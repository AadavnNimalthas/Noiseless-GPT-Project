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
            // ✅ Live camera feed
            CameraPreview(session: controller.session)
                .ignoresSafeArea()
            
            // ✅ Overlay dots/lines for each finger joint
            HandSkeletonOverlay(joints: controller.overlayJoints,
                                isMirrored: controller.overlayIsMirrored)
            .ignoresSafeArea()
            .allowsHitTesting(false)
            
            // Status UI
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

// MARK: - Hand skeleton overlay (dots for each finger joint)
struct HandSkeletonOverlay: View {
    /// Normalized (0...1) joint points from Vision
    let joints: [VNHumanHandPoseObservation.JointName: CGPoint]
    let isMirrored: Bool
    
    // Skeleton connections: Wrist -> each finger chain
    private let chains: [[VNHumanHandPoseObservation.JointName]] = [
        [.wrist, .thumbCMC, .thumbMP, .thumbIP, .thumbTip],
        [.wrist, .indexMCP, .indexPIP, .indexDIP, .indexTip],
        [.wrist, .middleMCP, .middlePIP, .middleDIP, .middleTip],
        [.wrist, .ringMCP, .ringPIP, .ringDIP, .ringTip],
        [.wrist, .littleMCP, .littlePIP, .littleDIP, .littleTip]
    ]
    
    var body: some View {
        GeometryReader { _ in
            Canvas { ctx, size in
                guard !joints.isEmpty else { return }
                
                func toView(_ p: CGPoint) -> CGPoint {
                    // Vision points are in normalized image space; flip to SwiftUI coords
                    let x = isMirrored ? (1.0 - p.x) : p.x
                    let y = 1.0 - p.y
                    return CGPoint(x: x * size.width, y: y * size.height)
                }
                
                // Draw finger lines
                for chain in chains {
                    for i in 0..<(chain.count - 1) {
                        let a = chain[i]
                        let b = chain[i + 1]
                        guard let paN = joints[a], let pbN = joints[b] else { continue }
                        
                        let pa = toView(paN)
                        let pb = toView(pbN)
                        
                        var path = Path()
                        path.move(to: pa)
                        path.addLine(to: pb)
                        ctx.stroke(path, with: .color(.white.opacity(0.75)), lineWidth: 2)
                    }
                }
                
                // ✅ Dots with thicker outlines + high-contrast color (not skin tone)
                let dotFill = Color.cyan.opacity(0.95)
                let dotStroke = Color.black.opacity(0.75)
                
                // Draw dots for every joint we have
                for (_, pN) in joints {
                    let p = toView(pN)
                    let r: CGFloat = 6
                    let rect = CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2)
                    let circle = Path(ellipseIn: rect)
                    
                    ctx.fill(circle, with: .color(dotFill))
                    ctx.stroke(circle, with: .color(dotStroke), lineWidth: 2.5)
                }
                
                // Make wrist a little bigger if present
                if let wristN = joints[.wrist] {
                    let p = toView(wristN)
                    let r: CGFloat = 9
                    let rect = CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2)
                    let circle = Path(ellipseIn: rect)
                    
                    ctx.fill(circle, with: .color(dotFill))
                    ctx.stroke(circle, with: .color(dotStroke), lineWidth: 3.0)
                }
            }
        }
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
    
    // Overlay data (per-joint dots)
    @Published var overlayJoints: [VNHumanHandPoseObservation.JointName: CGPoint] = [:]
    @Published var overlayIsMirrored: Bool = true
    
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
        let s = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        if s == .notDetermined {
            _ = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        }
    }
    
    // ✅ ONLY used for the BACK CAMERA photo orientation
    private func deviceToVideoOrientation() -> AVCaptureVideoOrientation {
        switch UIDevice.current.orientation {
        case .portrait:
            return .portrait
        case .portraitUpsideDown:
            return .portraitUpsideDown
        case .landscapeLeft:
            // Home indicator on the RIGHT
            return .landscapeRight
        case .landscapeRight:
            // Home indicator on the LEFT
            return .landscapeLeft
        default:
            // If faceUp/faceDown/unknown, default to portrait
            return .portrait
        }
    }
    
    private func applyBackPhotoOrientationNow() {
        guard currentPosition == .back else { return }
        let o = deviceToVideoOrientation()
        if let photoConn = photoOutput.connection(with: .video), photoConn.isVideoOrientationSupported {
            photoConn.videoOrientation = o
        }
    }
    
    private func configureSession(position: AVCaptureDevice.Position, enableGestureDetection: Bool) {
        session.beginConfiguration()
        session.sessionPreset = .photo
        
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
        
        if session.canAddOutput(photoOutput) {
            session.addOutput(photoOutput)
        }
        
        if enableGestureDetection {
            videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
            }
            // ✅ Keep front gesture pipeline exactly as before
            if let conn = videoOutput.connection(with: .video) {
                conn.videoOrientation = .portrait
            }
        }
        
        // ✅ Keep default as portrait (same as your working baseline)
        if let photoConn = photoOutput.connection(with: .video) {
            photoConn.videoOrientation = .portrait
        }
        
        currentPosition = position
        overlayIsMirrored = (position == .front)
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
            overlayJoints = [:]
        }
        
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
        
        // ✅ THIS is the change you wanted:
        // Set the BACK CAMERA photo orientation from how the iPad is currently held.
        applyBackPhotoOrientationNow()
        
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

// MARK: - Front camera frames -> gesture detection + overlay joints
extension FrontGestureBackCaptureController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        
        guard currentPosition == .front, !isReconfiguring else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Front camera image orientation (keep exactly as before)
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .leftMirrored,
                                            options: [:])
        
        do {
            try handler.perform([handPoseRequest])
            
            guard let obs = handPoseRequest.results?.first else {
                Task { @MainActor in
                    self.statusText = "Front cam: no hand…"
                    self.overlayJoints = [:]
                }
                resetTap()
                return
            }
            
            // Grab the points we want (all finger joints + wrist)
            let jointNames: [VNHumanHandPoseObservation.JointName] = [
                .wrist,
                .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
                .indexMCP, .indexPIP, .indexDIP, .indexTip,
                .middleMCP, .middlePIP, .middleDIP, .middleTip,
                .ringMCP, .ringPIP, .ringDIP, .ringTip,
                .littleMCP, .littlePIP, .littleDIP, .littleTip
            ]
            
            var j: [VNHumanHandPoseObservation.JointName: CGPoint] = [:]
            for name in jointNames {
                if let p = try? obs.recognizedPoint(name), p.confidence > 0.35 {
                    j[name] = p.location
                }
            }
            
            // Your existing gesture points
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
                self.overlayIsMirrored = true
                self.overlayJoints = j
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
