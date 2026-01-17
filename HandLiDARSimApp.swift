import SwiftUI
import RealityKit
import ARKit
import Vision
import simd
import UIKit
import ImageIO

// MARK: - App Entry
@main
struct HandLiDARSimApp: App {
    var body: some SwiftUI.Scene {
        WindowGroup {
            HandLiDARSimView()
        }
    }
}

struct HandLiDARSimView: View {
    @State private var showCameraFeed: Bool = true
    @State private var showSkeleton: Bool = true
    
    // ✅ Controls
    @State private var showMist: Bool = true
    @State private var freezeFrame: Bool = false
    @State private var showDebug: Bool = true
    
    @State private var smoothingBase: Float = 0.80
    @State private var handFadeSeconds: Float = 0.25
    @State private var minJointConfidence: Float = 0.35
    
    @State private var mistGlobalAlpha: Float = 0.65
    @State private var mistRadius: Float = 0.0028
    @State private var mistJitter: Float = 0.65
    @State private var mistBiasToJoints: Float = 0.65
    
    // ✅ Menu visibility (gesture + button)
    @State private var menuVisible: Bool = false
    
    var body: some View {
        ZStack(alignment: .topTrailing) {
            ARContainerView(showCameraFeed: $showCameraFeed,
                            showSkeleton: $showSkeleton,
                            showMist: $showMist,
                            freezeFrame: $freezeFrame,
                            showDebug: $showDebug,
                            smoothingBase: $smoothingBase,
                            handFadeSeconds: $handFadeSeconds,
                            minJointConfidence: $minJointConfidence,
                            mistGlobalAlpha: $mistGlobalAlpha,
                            mistRadius: $mistRadius,
                            mistJitter: $mistJitter,
                            mistBiasToJoints: $mistBiasToJoints,
                            menuVisible: $menuVisible)
            .ignoresSafeArea()
            
            // ✅ Right-side menu panel (not full-screen)
            if menuVisible {
                menuPanel
                    .transition(.move(edge: .trailing).combined(with: .opacity))
                    .padding(.top, 10)
                    .padding(.trailing, 10)
            }
            
            // ✅ Always-available small button (bottom-right)
            VStack {
                Spacer()
                HStack {
                    Spacer()
                    Button {
                        withAnimation(.spring(response: 0.25, dampingFraction: 0.9)) {
                            menuVisible.toggle()
                        }
                    } label: {
                        Image(systemName: menuVisible ? "xmark" : "slider.horizontal.3")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.white)
                            .padding(10)
                            .background(.black.opacity(0.55))
                            .clipShape(Circle())
                    }
                    .padding(.trailing, 14)
                    .padding(.bottom, 18)
                }
            }
        }
        .animation(.easeInOut(duration: 0.18), value: menuVisible)
    }
    
    private var menuPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Controls")
                    .font(.system(size: 14, weight: .semibold))
                Spacer()
                Button {
                    withAnimation(.spring(response: 0.25, dampingFraction: 0.9)) {
                        menuVisible = false
                    }
                } label: {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 14, weight: .semibold))
                }
            }
            
            Toggle("Show camera", isOn: $showCameraFeed).toggleStyle(.switch)
            Toggle("Show skeleton", isOn: $showSkeleton).toggleStyle(.switch)
            Toggle("Show mist", isOn: $showMist).toggleStyle(.switch)
            Toggle("Freeze frame", isOn: $freezeFrame).toggleStyle(.switch)
            Toggle("Debug (basic)", isOn: $showDebug).toggleStyle(.switch)
            
            Divider().opacity(0.25)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Smoothing: \(Int(smoothingBase * 100))%")
                    .font(.system(size: 12, weight: .semibold))
                Slider(value: $smoothingBase, in: 0.0...0.95)
                
                Text("Hand fade: \(String(format: "%.2f", handFadeSeconds))s")
                    .font(.system(size: 12, weight: .semibold))
                Slider(value: $handFadeSeconds, in: 0.05...0.75)
                
                Text("Min joint confidence: \(String(format: "%.2f", minJointConfidence))")
                    .font(.system(size: 12, weight: .semibold))
                Slider(value: $minJointConfidence, in: 0.10...0.75)
                
                Divider().opacity(0.25)
                
                Text("Mist alpha: \(String(format: "%.2f", mistGlobalAlpha))")
                    .font(.system(size: 12, weight: .semibold))
                Slider(value: $mistGlobalAlpha, in: 0.05...1.0)
                
                Text("Mist radius: \(String(format: "%.4f", mistRadius))")
                    .font(.system(size: 12, weight: .semibold))
                Slider(value: $mistRadius, in: 0.0012...0.0060)
                
                Text("Mist jitter: \(Int(mistJitter * 100))%")
                    .font(.system(size: 12, weight: .semibold))
                Slider(value: $mistJitter, in: 0.0...1.0)
                
                Text("Mist joint-bias: \(Int(mistBiasToJoints * 100))%")
                    .font(.system(size: 12, weight: .semibold))
                Slider(value: $mistBiasToJoints, in: 0.0...1.0)
            }
        }
        .padding(12)
        .frame(width: 290, alignment: .topLeading) // ✅ fixed width right panel
        .background(.ultraThinMaterial)
        .cornerRadius(14)
        .shadow(radius: 10)
    }
}

// MARK: - SwiftUI + RealityKit container
struct ARContainerView: UIViewRepresentable {
    @Binding var showCameraFeed: Bool
    @Binding var showSkeleton: Bool
    
    @Binding var showMist: Bool
    @Binding var freezeFrame: Bool
    @Binding var showDebug: Bool
    @Binding var smoothingBase: Float
    @Binding var handFadeSeconds: Float
    @Binding var minJointConfidence: Float
    @Binding var mistGlobalAlpha: Float
    @Binding var mistRadius: Float
    @Binding var mistJitter: Float
    @Binding var mistBiasToJoints: Float
    
    // ✅ Menu visible binding (gesture opens/closes)
    @Binding var menuVisible: Bool
    
    func makeCoordinator() -> Coordinator { Coordinator() }
    
    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        arView.automaticallyConfigureSession = false
        
        arView.environment.background = showCameraFeed ? .cameraFeed() : .color(.black)
        
        let config = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        config.planeDetection = [.horizontal, .vertical]
        
        arView.session.delegate = context.coordinator
        context.coordinator.attach(to: arView)
        
        context.coordinator.setSkeletonEnabled(showSkeleton)
        context.coordinator.setMistEnabled(showMist)
        
        // ✅ Provide menu visibility setters/getters to coordinator (so Vision swipe can control it)
        context.coordinator.setMenuController(
            get: { self.menuVisible },
            set: { newValue in
                DispatchQueue.main.async {
                    withAnimation(.spring(response: 0.25, dampingFraction: 0.9)) {
                        self.menuVisible = newValue
                    }
                }
            }
        )
        
        context.coordinator.updateSettings(
            freezeFrame: freezeFrame,
            showDebug: showDebug,
            smoothingBase: smoothingBase,
            handFadeSeconds: handFadeSeconds,
            minJointConfidence: minJointConfidence,
            mistGlobalAlpha: mistGlobalAlpha,
            mistRadius: mistRadius,
            mistJitter: mistJitter,
            mistBiasToJoints: mistBiasToJoints
        )
        
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        return arView
    }
    
    func updateUIView(_ arView: ARView, context: Context) {
        arView.environment.background = showCameraFeed ? .cameraFeed() : .color(.black)
        
        context.coordinator.setSkeletonEnabled(showSkeleton)
        context.coordinator.setMistEnabled(showMist)
        
        // keep menu controller closures fresh
        context.coordinator.setMenuController(
            get: { self.menuVisible },
            set: { newValue in
                DispatchQueue.main.async {
                    withAnimation(.spring(response: 0.25, dampingFraction: 0.9)) {
                        self.menuVisible = newValue
                    }
                }
            }
        )
        
        context.coordinator.updateSettings(
            freezeFrame: freezeFrame,
            showDebug: showDebug,
            smoothingBase: smoothingBase,
            handFadeSeconds: handFadeSeconds,
            minJointConfidence: minJointConfidence,
            mistGlobalAlpha: mistGlobalAlpha,
            mistRadius: mistRadius,
            mistJitter: mistJitter,
            mistBiasToJoints: mistBiasToJoints
        )
    }
}

// MARK: - Coordinator (ARSessionDelegate + Vision + Rendering)
final class Coordinator: NSObject, ARSessionDelegate {
    private weak var arView: ARView?
    private let cameraAnchor = AnchorEntity(.camera)
    
    private let skeleton = HandSkeletonRenderer()
    private let cloud = DepthCloudRenderer(maxPoints: 1200)
    
    private let handPoseRequest: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        r.maximumHandCount = 1
        return r
    }()
    
    private var lastVisionTime: TimeInterval = 0
    private let visionInterval: TimeInterval = 1.0 / 15.0
    
    // ✅ Settings
    private var freeze: Bool = false
    private var debugOn: Bool = true
    
    private var smoothingBase: Float = 0.80
    private var handFadeSeconds: Float = 0.25
    private var minJointConfidence: Float = 0.35
    
    private var mistGlobalAlpha: Float = 0.65
    private var mistRadius: Float = 0.0028
    private var mistJitter: Float = 0.65
    private var mistBiasToJoints: Float = 0.65
    
    // ✅ Toggles
    private var userSkeletonEnabled: Bool = true
    private var userMistEnabled: Bool = true
    
    // ✅ Smoothing/stability state
    private var smoothedJoints: [VNHumanHandPoseObservation.JointName: SIMD3<Float>] = [:]
    private var lastStableJoints: [VNHumanHandPoseObservation.JointName: SIMD3<Float>] = [:]
    
    // ✅ Hand fade
    private var handPresenceAlpha: Float = 0.0
    private var lastHandSeenTime: TimeInterval = 0
    
    // ✅ Menu controller closures (SwiftUI state bridge)
    private var menuGet: (() -> Bool)?
    private var menuSet: ((Bool) -> Void)?
    
    // ✅ Swipe detection state
    private var swipeLastCentroidX: CGFloat?
    private var swipeLastTime: TimeInterval = 0
    private var swipeAccumDX: CGFloat = 0
    private var swipeWindowStart: TimeInterval = 0
    private var lastMenuToggleTime: TimeInterval = 0
    
    func setMenuController(get: @escaping () -> Bool, set: @escaping (Bool) -> Void) {
        self.menuGet = get
        self.menuSet = set
    }
    
    func setSkeletonEnabled(_ enabled: Bool) {
        userSkeletonEnabled = enabled
    }
    
    func setMistEnabled(_ enabled: Bool) {
        userMistEnabled = enabled
    }
    
    func updateSettings(
        freezeFrame: Bool,
        showDebug: Bool,
        smoothingBase: Float,
        handFadeSeconds: Float,
        minJointConfidence: Float,
        mistGlobalAlpha: Float,
        mistRadius: Float,
        mistJitter: Float,
        mistBiasToJoints: Float
    ) {
        self.freeze = freezeFrame
        self.debugOn = showDebug
        
        self.smoothingBase = max(0.0, min(0.95, smoothingBase))
        self.handFadeSeconds = max(0.05, min(0.75, handFadeSeconds))
        self.minJointConfidence = max(0.10, min(0.75, minJointConfidence))
        
        self.mistGlobalAlpha = max(0.05, min(1.0, mistGlobalAlpha))
        self.mistRadius = max(0.0012, min(0.0060, mistRadius))
        self.mistJitter = max(0.0, min(1.0, mistJitter))
        self.mistBiasToJoints = max(0.0, min(1.0, mistBiasToJoints))
        
        cloud.setSphereRadius(self.mistRadius)
        cloud.setGlobalAlpha(self.mistGlobalAlpha)
    }
    
    func attach(to view: ARView) {
        self.arView = view
        view.scene.anchors.append(cameraAnchor)
        cameraAnchor.addChild(skeleton.root)
        cameraAnchor.addChild(cloud.root)
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let t = frame.timestamp
        guard t - lastVisionTime >= visionInterval else { return }
        lastVisionTime = t
        
        if freeze {
            applyHandFade(now: t, handDetectedThisFrame: false)
            return
        }
        
        guard let sceneDepth = frame.sceneDepth else {
            applyHandFade(now: t, handDetectedThisFrame: false)
            return
        }
        
        let captured = frame.capturedImage
        let depthMap = sceneDepth.depthMap
        let confMap  = sceneDepth.confidenceMap
        
        let orientation = Self.cgImageOrientationForCurrentDevice()
        let handler = VNImageRequestHandler(cvPixelBuffer: captured,
                                            orientation: orientation,
                                            options: [:])
        
        do {
            try handler.perform([handPoseRequest])
            guard let obs = handPoseRequest.results?.first else {
                applyHandFade(now: t, handDetectedThisFrame: false)
                resetSwipeIfNeeded(now: t)
                return
            }
            
            let jointNames: [VNHumanHandPoseObservation.JointName] = [
                .wrist,
                .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
                .indexMCP, .indexPIP, .indexDIP, .indexTip,
                .middleMCP, .middlePIP, .middleDIP, .middleTip,
                .ringMCP, .ringPIP, .ringDIP, .ringTip,
                .littleMCP, .littlePIP, .littleDIP, .littleTip
            ]
            
            var joints2D: [VNHumanHandPoseObservation.JointName: CGPoint] = [:]
            joints2D.reserveCapacity(jointNames.count)
            
            for jn in jointNames {
                if let p = try? obs.recognizedPoint(jn), p.confidence > minJointConfidence {
                    joints2D[jn] = p.location
                }
            }
            
            guard joints2D.count >= 8 else {
                applyHandFade(now: t, handDetectedThisFrame: false)
                resetSwipeIfNeeded(now: t)
                return
            }
            
            // ✅ Menu gesture (4-finger swipe left opens, right closes)
            detectMenuSwipe(joints2D: joints2D, now: t)
            
            let camera = frame.camera
            let imageRes = camera.imageResolution
            let intr = camera.intrinsics
            
            let joints3D = Self.liftJointsTo3D(joints2D: joints2D,
                                               depthMap: depthMap,
                                               imageResolution: imageRes,
                                               intrinsics: intr)
            
            guard joints3D.count >= 8 else {
                applyHandFade(now: t, handDetectedThisFrame: false)
                return
            }
            
            lastHandSeenTime = t
            applyHandFade(now: t, handDetectedThisFrame: true)
            
            let stable = stabilizeAndSmooth(joints3D)
            
            let skelVisible = userSkeletonEnabled && handPresenceAlpha > 0.01
            skeleton.setVisible(skelVisible)
            skeleton.setAlpha(handPresenceAlpha)
            if skelVisible {
                skeleton.update(joints3D: stable)
            }
            
            let mistVisible = userMistEnabled && handPresenceAlpha > 0.01
            cloud.setVisible(mistVisible)
            cloud.setPresenceAlpha(handPresenceAlpha)
            if mistVisible {
                cloud.updateAroundHand(joints2D: joints2D,
                                       depthMap: depthMap,
                                       confidenceMap: confMap,
                                       imageResolution: imageRes,
                                       intrinsics: intr,
                                       jitter: mistJitter,
                                       biasToJoints: mistBiasToJoints)
            }
        } catch {
            // ignore per-frame failures
        }
    }
    
    // MARK: - 4-finger swipe menu control
    
    private func detectMenuSwipe(joints2D: [VNHumanHandPoseObservation.JointName: CGPoint], now t: TimeInterval) {
        // Need index/middle/ring/little tips visible
        guard
            let i = joints2D[.indexTip],
            let m = joints2D[.middleTip],
            let r = joints2D[.ringTip],
            let l = joints2D[.littleTip]
        else {
            resetSwipeIfNeeded(now: t)
            return
        }
        
        // simple centroid in normalized image coords
        let cx = (i.x + m.x + r.x + l.x) / 4.0
        let cy = (i.y + m.y + r.y + l.y) / 4.0
        
        // (Optional) reject if points are too clustered (helps reduce accidental triggers)
        // If fingers are super close together, it might be a fist-ish shape.
        let spread = max(abs(i.x - l.x), abs(i.y - l.y))
        if spread < 0.06 {
            resetSwipeIfNeeded(now: t)
            return
        }
        
        // cooldown so it doesn't toggle repeatedly
        if t - lastMenuToggleTime < 0.60 {
            swipeLastCentroidX = cx
            swipeLastTime = t
            swipeWindowStart = t
            swipeAccumDX = 0
            _ = cy
            return
        }
        
        if swipeLastCentroidX == nil {
            swipeLastCentroidX = cx
            swipeLastTime = t
            swipeWindowStart = t
            swipeAccumDX = 0
            _ = cy
            return
        }
        
        let dt = max(0.0001, t - swipeLastTime)
        let dx = cx - (swipeLastCentroidX ?? cx)
        
        swipeAccumDX += dx
        swipeLastCentroidX = cx
        swipeLastTime = t
        
        // Keep a short time window for a "swipe"
        let window = t - swipeWindowStart
        if window > 0.35 {
            // reset the window (start a fresh swipe)
            swipeWindowStart = t
            swipeAccumDX = 0
        }
        
        // thresholds in normalized units
        // Left swipe => negative dx, Right swipe => positive dx
        let requiredDX: CGFloat = 0.18   // ~18% of screen width within the window
        let requiredV: CGFloat = 0.85    // normalized/sec-ish
        
        let v = CGFloat(dx) / CGFloat(dt)
        
        // OPEN (swipe left)
        if swipeAccumDX < -requiredDX && v < -requiredV {
            if (menuGet?() ?? false) == false {
                menuSet?(true)
                lastMenuToggleTime = t
            }
            swipeWindowStart = t
            swipeAccumDX = 0
        }
        
        // CLOSE (swipe right)
        if swipeAccumDX > requiredDX && v > requiredV {
            if (menuGet?() ?? false) == true {
                menuSet?(false)
                lastMenuToggleTime = t
            }
            swipeWindowStart = t
            swipeAccumDX = 0
        }
    }
    
    private func resetSwipeIfNeeded(now t: TimeInterval) {
        // if we lost the 4-finger signal for a bit, reset
        if t - swipeLastTime > 0.25 {
            swipeLastCentroidX = nil
            swipeAccumDX = 0
            swipeWindowStart = 0
        }
    }
    
    // MARK: - Fade logic
    private func applyHandFade(now t: TimeInterval, handDetectedThisFrame: Bool) {
        if handDetectedThisFrame {
            handPresenceAlpha = min(1.0, handPresenceAlpha + 0.22)
        } else {
            let dt = Float(visionInterval)
            let decayPerSecond = 1.0 / max(0.05, handFadeSeconds)
            handPresenceAlpha = max(0.0, handPresenceAlpha - decayPerSecond * dt)
        }
        
        let skelVisible = userSkeletonEnabled && handPresenceAlpha > 0.01
        skeleton.setVisible(skelVisible)
        skeleton.setAlpha(handPresenceAlpha)
        
        let mistVisible = userMistEnabled && handPresenceAlpha > 0.01
        cloud.setVisible(mistVisible)
        cloud.setPresenceAlpha(handPresenceAlpha)
        
        if handPresenceAlpha <= 0.01 {
            cloud.disableAllPoints()
        }
    }
    
    // MARK: - Stability
    private func stabilizeAndSmooth(_ new: [VNHumanHandPoseObservation.JointName: SIMD3<Float>])
    -> [VNHumanHandPoseObservation.JointName: SIMD3<Float>] {
        
        func smoothingFor(_ joint: VNHumanHandPoseObservation.JointName) -> Float {
            let tip: Set<VNHumanHandPoseObservation.JointName> = [.thumbTip, .indexTip, .middleTip, .ringTip, .littleTip]
            let dip: Set<VNHumanHandPoseObservation.JointName> = [.thumbIP, .indexDIP, .middleDIP, .ringDIP, .littleDIP]
            let pip: Set<VNHumanHandPoseObservation.JointName> = [.indexPIP, .middlePIP, .ringPIP, .littlePIP]
            let mcp: Set<VNHumanHandPoseObservation.JointName> = [.thumbMP, .thumbCMC, .indexMCP, .middleMCP, .ringMCP, .littleMCP]
            
            if joint == .wrist { return max(0.0, smoothingBase - 0.15) }
            if tip.contains(joint) { return min(0.95, smoothingBase + 0.10) }
            if dip.contains(joint) { return min(0.95, smoothingBase + 0.06) }
            if pip.contains(joint) { return min(0.95, smoothingBase + 0.03) }
            if mcp.contains(joint) { return smoothingBase }
            return smoothingBase
        }
        
        let maxJump: Float = 0.10
        
        var out: [VNHumanHandPoseObservation.JointName: SIMD3<Float>] = [:]
        out.reserveCapacity(new.count)
        
        for (k, vRaw) in new {
            let v: SIMD3<Float>
            if let prevStable = lastStableJoints[k] {
                let d = vRaw - prevStable
                let dist = simd_length(d)
                if dist > maxJump, dist.isFinite, dist > 0 {
                    v = prevStable + simd_normalize(d) * maxJump
                } else {
                    v = vRaw
                }
            } else {
                v = vRaw
            }
            
            let s = smoothingFor(k)
            
            if let prev = smoothedJoints[k] {
                let blended = prev * s + v * (1 - s)
                out[k] = blended
                smoothedJoints[k] = blended
                lastStableJoints[k] = blended
            } else {
                out[k] = v
                smoothedJoints[k] = v
                lastStableJoints[k] = v
            }
        }
        
        return out
    }
    
    // MARK: - Helpers
    private static func liftJointsTo3D(
        joints2D: [VNHumanHandPoseObservation.JointName: CGPoint],
        depthMap: CVPixelBuffer,
        imageResolution: CGSize,
        intrinsics: simd_float3x3
    ) -> [VNHumanHandPoseObservation.JointName: SIMD3<Float>] {
        
        let depthW = CVPixelBufferGetWidth(depthMap)
        let depthH = CVPixelBufferGetHeight(depthMap)
        
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        
        guard let depthBase = CVPixelBufferGetBaseAddress(depthMap) else { return [:] }
        let depthRowBytes = CVPixelBufferGetBytesPerRow(depthMap)
        
        let fx = intrinsics.columns.0.x
        let fy = intrinsics.columns.1.y
        let cx = intrinsics.columns.2.x
        let cy = intrinsics.columns.2.y
        
        var out: [VNHumanHandPoseObservation.JointName: SIMD3<Float>] = [:]
        out.reserveCapacity(joints2D.count)
        
        for (jn, pN) in joints2D {
            let u = Float(pN.x) * Float(imageResolution.width)
            let v = (1.0 - Float(pN.y)) * Float(imageResolution.height)
            
            let du = Int((u / Float(imageResolution.width)) * Float(depthW))
            let dv = Int((v / Float(imageResolution.height)) * Float(depthH))
            guard du >= 0, du < depthW, dv >= 0, dv < depthH else { continue }
            
            let rowPtr = depthBase.advanced(by: dv * depthRowBytes)
            let z = rowPtr.assumingMemoryBound(to: Float.self)[du]
            guard z.isFinite, z > 0.08, z < 2.5 else { continue }
            
            let x = (u - cx) / fx * z
            let y = -(v - cy) / fy * z
            out[jn] = SIMD3<Float>(x, y, z)
        }
        
        return out
    }
    
    private static func cgImageOrientationForCurrentDevice() -> CGImagePropertyOrientation {
        switch UIDevice.current.orientation {
        case .landscapeLeft: return .up
        case .landscapeRight: return .down
        case .portraitUpsideDown: return .left
        default: return .right
        }
    }
}

// MARK: - Hand skeleton renderer (bones + improved palm patch)
final class HandSkeletonRenderer {
    let root = Entity()
    private var boneEntities: [String: ModelEntity] = [:]
    private var palmEntity: ModelEntity?
    
    private let chains: [[VNHumanHandPoseObservation.JointName]] = [
        [.wrist, .thumbCMC, .thumbMP, .thumbIP, .thumbTip],
        [.wrist, .indexMCP, .indexPIP, .indexDIP, .indexTip],
        [.wrist, .middleMCP, .middlePIP, .middleDIP, .middleTip],
        [.wrist, .ringMCP, .ringPIP, .ringDIP, .ringTip],
        [.wrist, .littleMCP, .littlePIP, .littleDIP, .littleTip]
    ]
    
    private var currentAlpha: Float = 1.0
    
    private func boneMaterial(alpha: Float) -> SimpleMaterial {
        SimpleMaterial(color: .init(white: 0.58, alpha: CGFloat(0.95 * alpha)),
                       roughness: 0.35,
                       isMetallic: false)
    }
    
    private func palmMaterial(alpha: Float, facing: Float) -> SimpleMaterial {
        let base = 0.62
        let shaded = base + Double((facing - 0.5) * 0.10)
        return SimpleMaterial(color: .init(white: CGFloat(max(0.45, min(0.85, shaded))),
                                           alpha: CGFloat(0.55 * alpha)),
                              roughness: 0.65,
                              isMetallic: false)
    }
    
    func setVisible(_ visible: Bool) {
        root.isEnabled = visible
    }
    
    func setAlpha(_ a: Float) {
        currentAlpha = max(0.0, min(1.0, a))
        let mat = boneMaterial(alpha: currentAlpha)
        for (_, e) in boneEntities {
            e.model?.materials = [mat]
        }
    }
    
    func update(joints3D: [VNHumanHandPoseObservation.JointName: SIMD3<Float>]) {
        for chain in chains {
            for i in 0..<(chain.count - 1) {
                let a = chain[i]
                let b = chain[i + 1]
                guard let pa = joints3D[a], let pb = joints3D[b] else { continue }
                
                let key = "\(a.rawValue)-\(b.rawValue)"
                let bone = boneEntities[key] ?? makeBoneCylinder()
                boneEntities[key] = bone
                if bone.parent == nil { root.addChild(bone) }
                
                placeCylinder(bone,
                              from: SIMD3<Float>(pa.x, pa.y, -pa.z),
                              to:   SIMD3<Float>(pb.x, pb.y, -pb.z))
            }
        }
        
        updatePalm(joints3D: joints3D)
    }
    
    private func makeBoneCylinder() -> ModelEntity {
        let mesh = MeshResource.generateCylinder(height: 1.0, radius: 0.010)
        return ModelEntity(mesh: mesh, materials: [boneMaterial(alpha: currentAlpha)])
    }
    
    private func placeCylinder(_ e: ModelEntity, from a: SIMD3<Float>, to b: SIMD3<Float>) {
        let mid = (a + b) * 0.5
        let dir = b - a
        let len = max(simd_length(dir), 0.0001)
        
        e.position = mid
        
        let yAxis = SIMD3<Float>(0, 1, 0)
        let nDir = simd_normalize(dir)
        let axis = simd_cross(yAxis, nDir)
        let dot  = simd_dot(yAxis, nDir)
        let angle = acos(max(-1, min(1, dot)))
        
        if simd_length(axis) < 0.0001 {
            e.orientation = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
        } else {
            e.orientation = simd_quatf(angle: angle, axis: simd_normalize(axis))
        }
        
        let thickness: Float = (len > 0.045) ? 1.15 : 0.95
        e.scale = SIMD3<Float>(thickness, len, thickness)
    }
    
    private func updatePalm(joints3D: [VNHumanHandPoseObservation.JointName: SIMD3<Float>]) {
        guard
            let wrist = joints3D[.wrist],
            let thumbCMC = joints3D[.thumbCMC],
            let indexMCP = joints3D[.indexMCP],
            let middleMCP = joints3D[.middleMCP],
            let ringMCP = joints3D[.ringMCP],
            let littleMCP = joints3D[.littleMCP]
        else {
            palmEntity?.isEnabled = false
            return
        }
        
        let w = SIMD3<Float>(wrist.x, wrist.y, -wrist.z)
        let t = SIMD3<Float>(thumbCMC.x, thumbCMC.y, -thumbCMC.z)
        let i = SIMD3<Float>(indexMCP.x, indexMCP.y, -indexMCP.z)
        let m = SIMD3<Float>(middleMCP.x, middleMCP.y, -middleMCP.z)
        let r = SIMD3<Float>(ringMCP.x, ringMCP.y, -ringMCP.z)
        let l = SIMD3<Float>(littleMCP.x, littleMCP.y, -littleMCP.z)
        
        let positions: [SIMD3<Float>] = [w, t, i, m, r, l]
        
        let v1 = i - w
        let v2 = l - w
        let n = simd_normalize(simd_cross(v1, v2))
        let facing = max(0.0, min(1.0, (n.z + 1.0) * 0.5))
        
        let mat = palmMaterial(alpha: currentAlpha, facing: facing)
        
        if palmEntity == nil {
            palmEntity = ModelEntity(mesh: makePalmMeshFan(positions: positions),
                                     materials: [mat])
            if let palmEntity { root.addChild(palmEntity) }
        } else {
            palmEntity?.model?.mesh = makePalmMeshFan(positions: positions)
            palmEntity?.model?.materials = [mat]
        }
        
        palmEntity?.isEnabled = currentAlpha > 0.01
    }
    
    private func makePalmMeshFan(positions p: [SIMD3<Float>]) -> MeshResource {
        var indices: [UInt32] = []
        if p.count >= 4 {
            for k in 1..<(p.count - 1) {
                indices.append(0)
                indices.append(UInt32(k))
                indices.append(UInt32(k + 1))
            }
        }
        
        var desc = MeshDescriptor()
        desc.positions = MeshBuffers.Positions(p)
        desc.primitives = .triangles(indices)
        return try! MeshResource.generate(from: [desc])
    }
}

// MARK: - Depth cloud renderer
final class DepthCloudRenderer {
    let root = Entity()
    private var points: [ModelEntity] = []
    private let maxPoints: Int
    
    private var globalAlpha: Float = 0.65
    private var presenceAlpha: Float = 1.0
    private var sphereRadius: Float = 0.0028
    
    private var mats: [SimpleMaterial] = []
    private let bucketCount: Int = 10
    
    init(maxPoints: Int) {
        self.maxPoints = maxPoints
        points.reserveCapacity(maxPoints)
        
        rebuildMaterials()
        
        for _ in 0..<maxPoints {
            let e = ModelEntity(mesh: .generateSphere(radius: sphereRadius), materials: [mats[0]])
            e.isEnabled = false
            root.addChild(e)
            points.append(e)
        }
    }
    
    func setVisible(_ visible: Bool) {
        root.isEnabled = visible
    }
    
    func setGlobalAlpha(_ a: Float) {
        globalAlpha = max(0.05, min(1.0, a))
        rebuildMaterials()
        for p in points where p.isEnabled {
            p.model?.materials = [mats[0]]
        }
    }
    
    func setPresenceAlpha(_ a: Float) {
        presenceAlpha = max(0.0, min(1.0, a))
    }
    
    func setSphereRadius(_ r: Float) {
        let nr = max(0.0012, min(0.0060, r))
        guard abs(nr - sphereRadius) > 0.00001 else { return }
        sphereRadius = nr
        for p in points {
            p.model?.mesh = .generateSphere(radius: sphereRadius)
        }
    }
    
    func disableAllPoints() {
        for p in points { p.isEnabled = false }
    }
    
    private func rebuildMaterials() {
        mats.removeAll(keepingCapacity: true)
        mats.reserveCapacity(bucketCount)
        
        for b in 0..<bucketCount {
            let t = Float(b) / Float(max(1, bucketCount - 1))
            let white = CGFloat(0.85 - 0.35 * t)
            let alpha = CGFloat(globalAlpha)
            let m = SimpleMaterial(color: .init(white: white, alpha: alpha),
                                   roughness: 1.0,
                                   isMetallic: false)
            mats.append(m)
        }
    }
    
    private func confidenceMultiplier(_ c: UInt8?) -> Float {
        guard let c else { return 1.0 }
        switch c {
        case 2: return 1.00
        case 1: return 0.70
        default: return 0.40
        }
    }
    
    private func bucketIndex(forZ z: Float) -> Int {
        let zn = max(0.10, min(2.5, z))
        let t = (zn - 0.10) / (2.5 - 0.10)
        let idx = Int(round(t * Float(bucketCount - 1)))
        return max(0, min(bucketCount - 1, idx))
    }
    
    func updateAroundHand(
        joints2D: [VNHumanHandPoseObservation.JointName: CGPoint],
        depthMap: CVPixelBuffer,
        confidenceMap: CVPixelBuffer?,
        imageResolution: CGSize,
        intrinsics: simd_float3x3,
        jitter: Float,
        biasToJoints: Float
    ) {
        guard !joints2D.isEmpty else {
            for p in points { p.isEnabled = false }
            return
        }
        
        var minX: CGFloat = 1, minY: CGFloat = 1, maxX: CGFloat = 0, maxY: CGFloat = 0
        for (_, p) in joints2D {
            minX = min(minX, p.x); minY = min(minY, p.y)
            maxX = max(maxX, p.x); maxY = max(maxY, p.y)
        }
        
        let pad: CGFloat = 0.12
        minX = max(0, minX - pad); minY = max(0, minY - pad)
        maxX = min(1, maxX + pad); maxY = min(1, maxY + pad)
        
        let depthW = CVPixelBufferGetWidth(depthMap)
        let depthH = CVPixelBufferGetHeight(depthMap)
        
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        
        guard let depthBase = CVPixelBufferGetBaseAddress(depthMap) else {
            for p in points { p.isEnabled = false }
            return
        }
        
        let depthRowBytes = CVPixelBufferGetBytesPerRow(depthMap)
        
        var confBase: UnsafeMutableRawPointer?
        var confRowBytes: Int = 0
        var confW: Int = 0
        var confH: Int = 0
        
        if let confidenceMap {
            confW = CVPixelBufferGetWidth(confidenceMap)
            confH = CVPixelBufferGetHeight(confidenceMap)
            CVPixelBufferLockBaseAddress(confidenceMap, .readOnly)
            confBase = CVPixelBufferGetBaseAddress(confidenceMap)
            confRowBytes = CVPixelBufferGetBytesPerRow(confidenceMap)
            defer { CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly) }
        }
        
        let fx = intrinsics.columns.0.x
        let fy = intrinsics.columns.1.y
        let cx = intrinsics.columns.2.x
        let cy = intrinsics.columns.2.y
        
        let stepsX = 40
        let stepsY = 30
        
        let jointCount = joints2D.count
        let extraNearJoints = Int(Float(maxPoints) * 0.20 * max(0.0, min(1.0, biasToJoints)))
        let useNearJointSamples = extraNearJoints > 0 && jointCount > 0
        let jointArray: [CGPoint] = Array(joints2D.values)
        
        var idx = 0
        
        if useNearJointSamples {
            for s in 0..<extraNearJoints {
                if idx >= maxPoints { break }
                let jp = jointArray[s % jointArray.count]
                
                let r: CGFloat = 0.018 + CGFloat.random(in: 0...0.020)
                let ang = CGFloat.random(in: 0..<(2 * .pi))
                let ox = cos(ang) * r
                let oy = sin(ang) * r
                
                let nx = max(minX, min(maxX, jp.x + ox))
                let ny = max(minY, min(maxY, jp.y + oy))
                
                _ = placePoint(atNX: nx, ny: ny,
                               idx: &idx,
                               depthBase: depthBase, depthRowBytes: depthRowBytes, depthW: depthW, depthH: depthH,
                               confBase: confBase, confRowBytes: confRowBytes, confW: confW, confH: confH,
                               imageResolution: imageResolution,
                               fx: fx, fy: fy, cx: cx, cy: cy)
            }
        }
        
        for gy in 0..<stepsY {
            for gx in 0..<stepsX {
                if idx >= maxPoints { break }
                
                let baseNX = minX + (CGFloat(gx) / CGFloat(stepsX - 1)) * (maxX - minX)
                let baseNY = minY + (CGFloat(gy) / CGFloat(stepsY - 1)) * (maxY - minY)
                
                let cellW = (maxX - minX) / CGFloat(max(1, stepsX - 1))
                let cellH = (maxY - minY) / CGFloat(max(1, stepsY - 1))
                
                let jx = (CGFloat.random(in: -0.5...0.5) * cellW) * CGFloat(jitter)
                let jy = (CGFloat.random(in: -0.5...0.5) * cellH) * CGFloat(jitter)
                
                let nx = max(minX, min(maxX, baseNX + jx))
                let ny = max(minY, min(maxY, baseNY + jy))
                
                _ = placePoint(atNX: nx, ny: ny,
                               idx: &idx,
                               depthBase: depthBase, depthRowBytes: depthRowBytes, depthW: depthW, depthH: depthH,
                               confBase: confBase, confRowBytes: confRowBytes, confW: confW, confH: confH,
                               imageResolution: imageResolution,
                               fx: fx, fy: fy, cx: cx, cy: cy)
            }
            if idx >= maxPoints { break }
        }
        
        while idx < maxPoints {
            points[idx].isEnabled = false
            idx += 1
        }
    }
    
    private func placePoint(
        atNX nx: CGFloat,
        ny: CGFloat,
        idx: inout Int,
        depthBase: UnsafeMutableRawPointer,
        depthRowBytes: Int,
        depthW: Int,
        depthH: Int,
        confBase: UnsafeMutableRawPointer?,
        confRowBytes: Int,
        confW: Int,
        confH: Int,
        imageResolution: CGSize,
        fx: Float, fy: Float, cx: Float, cy: Float
    ) -> Bool {
        let u = Float(nx) * Float(imageResolution.width)
        let v = (1.0 - Float(ny)) * Float(imageResolution.height)
        
        let du = Int((u / Float(imageResolution.width)) * Float(depthW))
        let dv = Int((v / Float(imageResolution.height)) * Float(depthH))
        guard du >= 0, du < depthW, dv >= 0, dv < depthH else { return false }
        
        var confValue: UInt8?
        if let confBase {
            let cu = Int((u / Float(imageResolution.width)) * Float(confW))
            let cv = Int((v / Float(imageResolution.height)) * Float(confH))
            guard cu >= 0, cu < confW, cv >= 0, cv < confH else { return false }
            
            let cRow = confBase.advanced(by: cv * confRowBytes)
            let conf = cRow.assumingMemoryBound(to: UInt8.self)[cu]
            confValue = conf
            guard conf >= 1 else { return false }
        }
        
        let dRow = depthBase.advanced(by: dv * depthRowBytes)
        let z = dRow.assumingMemoryBound(to: Float.self)[du]
        guard z.isFinite, z > 0.10, z < 2.5 else { return false }
        
        let x = (u - cx) / fx * z
        let y = -(v - cy) / fy * z
        
        let e = points[idx]
        e.isEnabled = true
        e.position = SIMD3<Float>(x, y, -z)
        
        let b = bucketIndex(forZ: z)
        let confMul = confidenceMultiplier(confValue)
        let finalAlpha = max(0.0, min(1.0, globalAlpha * confMul * presenceAlpha))
        
        let approxBucketAlpha: Float = globalAlpha
        if abs(finalAlpha - approxBucketAlpha) > 0.10 {
            let white = CGFloat(0.85 - 0.35 * (Float(b) / Float(max(1, bucketCount - 1))))
            let m = SimpleMaterial(color: .init(white: white, alpha: CGFloat(finalAlpha)),
                                   roughness: 1.0,
                                   isMetallic: false)
            e.model?.materials = [m]
        } else {
            e.model?.materials = [mats[b]]
        }
        
        idx += 1
        return true
    }
}
