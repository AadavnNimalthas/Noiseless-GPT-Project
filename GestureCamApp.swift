import SwiftUI
import AVFoundation
import Vision
import Photos
import UIKit

@main
struct GestureCamApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @StateObject private var controller = FrontGestureBackCaptureController()

    var body: some View {
        ZStack {
            CameraPreview(session: controller.session)
                .ignoresSafeArea()

            VStack {
                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(controller.statusText)
                            .font(.system(size: 14, weight: .semibold))
                        Text("Gesture: fist + thumb-index tap x2")
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                        if !controller.lastSavedMessage.isEmpty {
                            Text(controller.lastSavedMessage)
                                .font(.system(size: 12, weight: .semibold))
                        }
                    }
                    .padding(10)
                    .background(.ultraThinMaterial)
                    .cornerRadius(12)

                    Spacer()
                }
                .padding()

                Spacer()
            }
        }
        .onAppear { controller.start() }
        .onDisappear { controller.stop() }
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(layer)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        guard let layer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer else { return }
        layer.session = session
        layer.frame = uiView.bounds
    }
}

@MainActor
final class FrontGestureBackCaptureController: NSObject, ObservableObject {
    // Camera session
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let photoOutput = AVCapturePhotoOutput()
    private let videoQueue = DispatchQueue(label: "front.video.queue")

    // State
    @Published var statusText: String = "Starting…"
    @Published var lastSavedMessage: String = ""

    private var currentPosition: AVCaptureDevice.Position = .front
    private var isReconfiguring = false

    // Gesture tracking
    private var lastPinchState = false
    private var tapCount = 0
    private var firstTapTime: CFTimeInterval = 0
    private var lastTriggerTime: CFTimeInterval = 0

    // Tune these
    private let pinchThreshold: CGFloat = 0.06
    private let tapWindow: CFTimeInterval = 1.0
    private let triggerCooldown: CFTimeInterval = 2.0

    // Vision
    private let handPoseRequest: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        r.maximumHandCount = 1
        return r
    }()

    func start() {
        Task {
            await requestPermissions()
            configureSession(position: .front, enableGestureDetection: true)
            session.startRunning()
            statusText = "Front cam: looking for gesture…"
        }
    }

    func stop() {
        session.stopRunning()
    }

    private func requestPermissions() async {
        // Photos Add permission
        let s = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        if s == .notDetermined {
            _ = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        }
        // Camera permission will be requested automatically when session starts
    }

    private func configureSession(position: AVCaptureDevice.Position, enableGestureDetection: Bool) {
        session.beginConfiguration()
        session.sessionPreset = .photo

        // Clear existing
        for input in session.inputs { session.removeInput(input) }
        for output in session.outputs { session.removeOutput(output) }

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input)
        else {
            session.commitConfiguration()
            statusText = "Camera unavailable."
            return
        }
        session.addInput(input)

        // Photo output always on (so we can capture on back)
        if session.canAddOutput(photoOutput) {
            session.addOutput(photoOutput)
        }

        // Video output only when we want gesture detection (front)
        if enableGestureDetection {
            videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)

            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
            }
            if let conn = videoOutput.connection(with: .video) {
                conn.videoOrientation = .portrait
            }
        }

        // Orientation for photo connection
        if let photoConn = photoOutput.connection(with: .video) {
            photoConn.videoOrientation = .portrait
        }

        currentPosition = position
        session.commitConfiguration()
    }

    private func triggerBackCapture() {
        let now = CACurrentMediaTime()
        guard now - lastTriggerTime > triggerCooldown else { return }
        lastTriggerTime = now

        guard !isReconfiguring else { return }
        isReconfiguring = true

        Task { @MainActor in
            statusText = "Switching to back camera…"
            lastSavedMessage = ""
        }

        // Switch to back cam, capture, then switch back
        DispatchQueue.main.async {
            self.session.stopRunning()
            self.configureSession(position: .back, enableGestureDetection: false)
            self.session.startRunning()

            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                self.capturePhoto()
            }
        }
    }

    private func capturePhoto() {
        Task { @MainActor in self.statusText = "Capturing… (back camera)" }

        let settings = AVCapturePhotoSettings()
        settings.flashMode = .off
        photoOutput.capturePhoto(with: settings, delegate: self)
    }

    private func switchBackToFrontAfterCapture() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            self.session.stopRunning()
            self.configureSession(position: .front, enableGestureDetection: true)
            self.session.startRunning()
            Task { @MainActor in
                self.statusText = "Front cam: looking for gesture…"
                self.isReconfiguring = false
            }
        }
    }

    private func resetTap() {
        tapCount = 0
        firstTapTime = 0
        lastPinchState = false
    }
}

// MARK: - Front camera frames -> gesture detection
extension FrontGestureBackCaptureController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {

        // Only detect gesture on front cam mode
        guard currentPosition == .front, !isReconfiguring else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Front camera orientation
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .leftMirrored,
                                            options: [:])
        do {
            try handler.perform([handPoseRequest])
            guard let obs = handPoseRequest.results?.first else {
                Task { @MainActor in self.statusText = "Front cam: no hand…" }
                resetTap()
                return
            }

            let thumb = try obs.recognizedPoint(.thumbTip)
            let index = try obs.recognizedPoint(.indexTip)
            let wrist = try obs.recognizedPoint(.wrist)
            let middle = try obs.recognizedPoint(.middleTip)
            let ring = try obs.recognizedPoint(.ringTip)
            let little = try obs.recognizedPoint(.littleTip)

            guard thumb.confidence > 0.5, index.confidence > 0.5, wrist.confidence > 0.2 else {
                return
            }

            let pinchDist = hypot(thumb.location.x - index.location.x,
                                 thumb.location.y - index.location.y)
            let isPinching = pinchDist < pinchThreshold
            let isFist = fistHeuristic(wrist: wrist, tips: [middle, ring, little]) > 0

            Task { @MainActor in
                self.statusText = "Front: Fist \(isFist ? "✅" : "—")  Pinch \(isPinching ? "✅" : "—")"
            }

            processTap(isFist: isFist, isPinching: isPinching)
        } catch {
            // ignore per-frame failures
        }
    }

    private func fistHeuristic(wrist: VNRecognizedPoint, tips: [VNRecognizedPoint]) -> CGFloat {
        let confident = tips.filter { $0.confidence > 0.3 }
        guard confident.count >= 2 else { return -1 }

        let dists = confident.map { hypot($0.location.x - wrist.location.x,
                                         $0.location.y - wrist.location.y) }
        let avg = dists.reduce(0, +) / CGFloat(dists.count)
        let spread = (dists.max() ?? avg) - (dists.min() ?? avg)

        // Tune for your lighting/distance
        return (avg < 0.28 && spread < 0.10) ? 1 : -1
    }

    private func processTap(isFist: Bool, isPinching: Bool) {
        let now = CACurrentMediaTime()

        if !isFist {
            resetTap()
            lastPinchState = isPinching
            return
        }

        // Rising edge pinch counts as a "tap"
        if isPinching && !lastPinchState {
            if tapCount == 0 {
                firstTapTime = now
                tapCount = 1
            } else {
                if now - firstTapTime <= tapWindow {
                    tapCount += 1
                } else {
                    firstTapTime = now
                    tapCount = 1
                }
            }

            if tapCount >= 2 {
                resetTap()
                triggerBackCapture()
            }
        }

        // Window expired
        if tapCount > 0 && now - firstTapTime > tapWindow {
            resetTap()
        }

        lastPinchState = isPinching
    }
}

// MARK: - Photo capture -> save to Photos
extension FrontGestureBackCaptureController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {

        if let error {
            Task { @MainActor in
                self.statusText = "Capture error: \(error.localizedDescription)"
                self.isReconfiguring = false
            }
            switchBackToFrontAfterCapture()
            return
        }

        guard let data = photo.fileDataRepresentation(),
              let image = UIImage(data: data) else {
            Task { @MainActor in
                self.statusText = "Failed to build image."
                self.isReconfiguring = false
            }
            switchBackToFrontAfterCapture()
            return
        }

        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.creationRequestForAsset(from: image)
        }, completionHandler: { success, err in
            Task { @MainActor in
                if success {
                    self.lastSavedMessage = "Saved to Photos ✅"
                    self.statusText = "Saved ✅ (switching back)"
                } else {
                    self.lastSavedMessage = "Save failed: \(err?.localizedDescription ?? "Unknown")"
                    self.statusText = "Save failed (switching back)"
                }
            }
            self.switchBackToFrontAfterCapture()
        })
    }
}
