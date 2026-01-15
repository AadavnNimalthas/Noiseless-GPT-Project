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
    // ✅ Fix: RealityKit also has a Scene type; force SwiftUI.Scene
    var body: some SwiftUI.Scene {
        WindowGroup {
            HandLiDARSimView()
        }
    }
}

struct HandLiDARSimView: View {
    var body: some View {
        ARContainerView()
            .ignoresSafeArea()
    }
}

// MARK: - SwiftUI + RealityKit container
struct ARContainerView: UIViewRepresentable {
    func makeCoordinator() -> Coordinator { Coordinator() }
    
    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        arView.automaticallyConfigureSession = false
        arView.environment.background = .cameraFeed()
        
        // Session config
        let config = ARWorldTrackingConfiguration()
        
        // ✅ Enable LiDAR scene depth if supported
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        
        // Optional: helps tracking stability in rooms
        config.planeDetection = [.horizontal, .vertical]
        
        arView.session.delegate = context.coordinator
        context.coordinator.attach(to: arView)
        
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {}
}

// MARK: - Coordinator (ARSessionDelegate + Vision + Rendering)
final class Coordinator: NSObject, ARSessionDelegate {
    // RealityKit
    private weak var arView: ARView?
    private let cameraAnchor = AnchorEntity(.camera)
    
    // Renderers
    private let skeleton = HandSkeletonRenderer()
    private let cloud = DepthCloudRenderer(maxPoints: 180)
    
    // Vision
    private let handPoseRequest: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        r.maximumHandCount = 1
        return r
    }()
    
    // Throttle Vision (keeps fps smooth)
    private var lastVisionTime: TimeInterval = 0
    private let visionInterval: TimeInterval = 1.0 / 15.0 // 15 Hz
    
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
        
        // Depth (LiDAR) – may be nil if unsupported or not enabled
        guard let sceneDepth = frame.sceneDepth else {
            skeleton.setVisible(false)
            cloud.setVisible(false)
            return
        }
        
        let captured = frame.capturedImage
        let depthMap = sceneDepth.depthMap                  // CVPixelBuffer
        let confMap  = sceneDepth.confidenceMap             // CVPixelBuffer? (optional)
        
        // Run Vision on captured image
        let orientation = Self.cgImageOrientationForCurrentDevice()
        let handler = VNImageRequestHandler(cvPixelBuffer: captured,
                                            orientation: orientation,
                                            options: [:])
        
        do {
            try handler.perform([handPoseRequest])
            
            guard let obs = handPoseRequest.results?.first else {
                skeleton.setVisible(false)
                cloud.setVisible(false)
                return
            }
            
            // Joint points (normalized 0..1)
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
                if let p = try? obs.recognizedPoint(jn), p.confidence > 0.35 {
                    joints2D[jn] = p.location
                }
            }
            
            guard joints2D.count >= 8 else {
                skeleton.setVisible(false)
                cloud.setVisible(false)
                return
            }
            
            // Convert joints into 3D camera space using depth + intrinsics
            let camera = frame.camera
            let imageRes = camera.imageResolution
            let intr = camera.intrinsics
            
            let joints3D = Self.liftJointsTo3D(joints2D: joints2D,
                                               depthMap: depthMap,
                                               imageResolution: imageRes,
                                               intrinsics: intr)
            
            guard joints3D.count >= 8 else {
                skeleton.setVisible(false)
                cloud.setVisible(false)
                return
            }
            
            skeleton.setVisible(true)
            skeleton.update(joints3D: joints3D)
            
            // Depth point cloud around hand ROI (“mist”)
            cloud.setVisible(true)
            cloud.updateAroundHand(joints2D: joints2D,
                                   depthMap: depthMap,
                                   confidenceMap: confMap,
                                   imageResolution: imageRes,
                                   intrinsics: intr)
        } catch {
            // Ignore per-frame Vision errors to keep AR smooth
        }
    }
    
    // MARK: - Helpers
    
    /// Lift Vision 2D joints (normalized) into camera-space 3D using depth + intrinsics.
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
            // Vision normalized coords (origin lower-left)
            let u = Float(pN.x) * Float(imageResolution.width)
            let v = (1.0 - Float(pN.y)) * Float(imageResolution.height)
            
            // Map to depth pixel coords (depthMap is lower-res)
            let du = Int((u / Float(imageResolution.width)) * Float(depthW))
            let dv = Int((v / Float(imageResolution.height)) * Float(depthH))
            
            guard du >= 0, du < depthW, dv >= 0, dv < depthH else { continue }
            
            let rowPtr = depthBase.advanced(by: dv * depthRowBytes)
            let z = rowPtr.assumingMemoryBound(to: Float.self)[du]
            guard z.isFinite, z > 0.08, z < 2.5 else { continue }
            
            // Unproject to camera space
            let x = (u - cx) / fx * z
            let y = (v - cy) / fy * z
            out[jn] = SIMD3<Float>(x, y, z)
        }
        
        return out
    }
    
    /// Best-effort orientation mapping for Vision
    private static func cgImageOrientationForCurrentDevice() -> CGImagePropertyOrientation {
        switch UIDevice.current.orientation {
        case .landscapeLeft:
            return .up
        case .landscapeRight:
            return .down
        case .portraitUpsideDown:
            return .left
        default:
            return .right
        }
    }
}

// MARK: - Hand skeleton renderer (grey spheres + cylinders)
final class HandSkeletonRenderer {
    let root = Entity()
    
    private var jointEntities: [VNHumanHandPoseObservation.JointName: ModelEntity] = [:]
    private var boneEntities: [String: ModelEntity] = [:]
    
    private let chains: [[VNHumanHandPoseObservation.JointName]] = [
        [.wrist, .thumbCMC, .thumbMP, .thumbIP, .thumbTip],
        [.wrist, .indexMCP, .indexPIP, .indexDIP, .indexTip],
        [.wrist, .middleMCP, .middlePIP, .middleDIP, .middleTip],
        [.wrist, .ringMCP, .ringPIP, .ringDIP, .ringTip],
        [.wrist, .littleMCP, .littlePIP, .littleDIP, .littleTip]
    ]
    
    private let jointMat = SimpleMaterial(color: .init(white: 0.78, alpha: 1.0), roughness: 0.9, isMetallic: false)
    private let boneMat  = SimpleMaterial(color: .init(white: 0.55, alpha: 1.0), roughness: 0.95, isMetallic: false)
    
    func setVisible(_ visible: Bool) {
        root.isEnabled = visible
    }
    
    func update(joints3D: [VNHumanHandPoseObservation.JointName: SIMD3<Float>]) {
        // Joints
        for (jn, p) in joints3D {
            let e = jointEntities[jn] ?? makeJointSphere(radius: jn == .wrist ? 0.012 : 0.008)
            jointEntities[jn] = e
            if e.parent == nil { root.addChild(e) }
            
            // AnchorEntity(.camera) is camera-local; negative z is "in front"
            e.position = SIMD3<Float>(p.x, p.y, -p.z)
        }
        
        // Bones
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
    }
    
    private func makeJointSphere(radius: Float) -> ModelEntity {
        ModelEntity(mesh: .generateSphere(radius: radius), materials: [jointMat])
    }
    
    private func makeBoneCylinder() -> ModelEntity {
        // Unit cylinder aligned to Y; we scale Y to bone length
        let mesh = MeshResource.generateCylinder(height: 1.0, radius: 0.004)
        return ModelEntity(mesh: mesh, materials: [boneMat])
    }
    
    private func placeCylinder(_ e: ModelEntity, from a: SIMD3<Float>, to b: SIMD3<Float>) {
        let mid = (a + b) * 0.5
        let dir = b - a
        let len = max(simd_length(dir), 0.0001)
        
        e.position = mid
        
        // Rotate +Y to align with dir
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
        
        // Scale height to length
        e.scale = SIMD3<Float>(1, len, 1)
    }
}

// MARK: - Depth cloud renderer (grey "mist" around hand)
final class DepthCloudRenderer {
    let root = Entity()
    private var points: [ModelEntity] = []
    private let maxPoints: Int
    
    private let mat = SimpleMaterial(color: .init(white: 0.70, alpha: 0.9), roughness: 1.0, isMetallic: false)
    
    init(maxPoints: Int) {
        self.maxPoints = maxPoints
        points.reserveCapacity(maxPoints)
        
        for _ in 0..<maxPoints {
            let e = ModelEntity(mesh: .generateSphere(radius: 0.0045), materials: [mat])
            e.isEnabled = false
            root.addChild(e)
            points.append(e)
        }
    }
    
    func setVisible(_ visible: Bool) {
        root.isEnabled = visible
    }
    
    func updateAroundHand(
        joints2D: [VNHumanHandPoseObservation.JointName: CGPoint],
        depthMap: CVPixelBuffer,
        confidenceMap: CVPixelBuffer?,
        imageResolution: CGSize,
        intrinsics: simd_float3x3
    ) {
        guard !joints2D.isEmpty else {
            for p in points { p.isEnabled = false }
            return
        }
        
        // ROI in normalized coords (expand a bit)
        var minX: CGFloat = 1, minY: CGFloat = 1, maxX: CGFloat = 0, maxY: CGFloat = 0
        for (_, p) in joints2D {
            minX = min(minX, p.x); minY = min(minY, p.y)
            maxX = max(maxX, p.x); maxY = max(maxY, p.y)
        }
        
        let pad: CGFloat = 0.10
        minX = max(0, minX - pad); minY = max(0, minY - pad)
        maxX = min(1, maxX + pad); maxY = min(1, maxY + pad)
        
        // Depth dims
        let depthW = CVPixelBufferGetWidth(depthMap)
        let depthH = CVPixelBufferGetHeight(depthMap)
        
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        
        guard let depthBase = CVPixelBufferGetBaseAddress(depthMap) else {
            for p in points { p.isEnabled = false }
            return
        }
        
        let depthRowBytes = CVPixelBufferGetBytesPerRow(depthMap)
        
        // Optional confidence
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
        
        // Grid sampling inside ROI (lightweight)
        let stepsX = 18
        let stepsY = 10
        
        var idx = 0
        for gy in 0..<stepsY {
            for gx in 0..<stepsX {
                if idx >= maxPoints { break }
                
                let nx = minX + (CGFloat(gx) / CGFloat(stepsX - 1)) * (maxX - minX)
                let ny = minY + (CGFloat(gy) / CGFloat(stepsY - 1)) * (maxY - minY)
                
                let u = Float(nx) * Float(imageResolution.width)
                let v = (1.0 - Float(ny)) * Float(imageResolution.height)
                
                let du = Int((u / Float(imageResolution.width)) * Float(depthW))
                let dv = Int((v / Float(imageResolution.height)) * Float(depthH))
                guard du >= 0, du < depthW, dv >= 0, dv < depthH else { continue }
                
                // Confidence filter (if available): keep medium/high only
                if let confBase {
                    let cu = Int((u / Float(imageResolution.width)) * Float(confW))
                    let cv = Int((v / Float(imageResolution.height)) * Float(confH))
                    guard cu >= 0, cu < confW, cv >= 0, cv < confH else { continue }
                    
                    let cRow = confBase.advanced(by: cv * confRowBytes)
                    let conf = cRow.assumingMemoryBound(to: UInt8.self)[cu]
                    guard conf >= 1 else { continue }
                }
                
                let dRow = depthBase.advanced(by: dv * depthRowBytes)
                let z = dRow.assumingMemoryBound(to: Float.self)[du]
                guard z.isFinite, z > 0.10, z < 2.5 else { continue }
                
                // Unproject
                let x = (u - cx) / fx * z
                let y = (v - cy) / fy * z
                
                let e = points[idx]
                e.isEnabled = true
                e.position = SIMD3<Float>(x, y, -z)
                idx += 1
            }
            if idx >= maxPoints { break }
        }
        
        // Disable leftover points
        while idx < maxPoints {
            points[idx].isEnabled = false
            idx += 1
        }
    }
}
