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
    
    var body: some View {
        ZStack(alignment: .topLeading) {
            ARContainerView(showCameraFeed: $showCameraFeed,
                            showSkeleton: $showSkeleton)
            .ignoresSafeArea()
            
            VStack(alignment: .leading, spacing: 10) {
                Toggle("Show camera", isOn: $showCameraFeed)
                    .toggleStyle(.switch)
                
                Toggle("Show skeleton", isOn: $showSkeleton)
                    .toggleStyle(.switch)
            }
            .padding(12)
            .background(.ultraThinMaterial)
            .cornerRadius(12)
            .padding()
        }
    }
}

// MARK: - SwiftUI + RealityKit container
struct ARContainerView: UIViewRepresentable {
    @Binding var showCameraFeed: Bool
    @Binding var showSkeleton: Bool
    
    func makeCoordinator() -> Coordinator { Coordinator() }
    
    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        arView.automaticallyConfigureSession = false
        
        // Background starts based on toggle
        arView.environment.background = showCameraFeed ? .cameraFeed() : .color(.black)
        
        let config = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        config.planeDetection = [.horizontal, .vertical]
        
        arView.session.delegate = context.coordinator
        context.coordinator.attach(to: arView)
        
        // Initial toggle states
        context.coordinator.setSkeletonEnabled(showSkeleton)
        
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        return arView
    }
    
    func updateUIView(_ arView: ARView, context: Context) {
        // Toggle camera feed on/off
        arView.environment.background = showCameraFeed ? .cameraFeed() : .color(.black)
        
        // Toggle skeleton on/off
        context.coordinator.setSkeletonEnabled(showSkeleton)
    }
}

// MARK: - Coordinator (ARSessionDelegate + Vision + Rendering)
final class Coordinator: NSObject, ARSessionDelegate {
    private weak var arView: ARView?
    private let cameraAnchor = AnchorEntity(.camera)
    
    private let skeleton = HandSkeletonRenderer()
    private let cloud = DepthCloudRenderer(maxPoints: 1200) // denser "hand surface" feel
    
    private let handPoseRequest: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        r.maximumHandCount = 1
        return r
    }()
    
    private var lastVisionTime: TimeInterval = 0
    private let visionInterval: TimeInterval = 1.0 / 15.0 // 15 Hz
    
    // Smoothing state
    private var smoothedJoints: [VNHumanHandPoseObservation.JointName: SIMD3<Float>] = [:]
    private let smoothing: Float = 0.80
    
    // ✅ User toggle state (toggle should "win")
    private var userSkeletonEnabled: Bool = true
    
    func setSkeletonEnabled(_ enabled: Bool) {
        userSkeletonEnabled = enabled
        skeleton.setVisible(enabled)
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
        
        guard let sceneDepth = frame.sceneDepth else {
            skeleton.setVisible(false)  // no depth -> hide
            cloud.setVisible(false)
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
                skeleton.setVisible(false)  // no hand -> hide regardless of toggle
                cloud.setVisible(false)
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
                if let p = try? obs.recognizedPoint(jn), p.confidence > 0.35 {
                    joints2D[jn] = p.location
                }
            }
            
            guard joints2D.count >= 8 else {
                skeleton.setVisible(false)
                cloud.setVisible(false)
                return
            }
            
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
            
            // Smooth the 3D joints
            let stable = smoothJoints(joints3D)
            
            // ✅ Toggle wins: only show if user enabled
            skeleton.setVisible(userSkeletonEnabled)
            if userSkeletonEnabled {
                skeleton.update(joints3D: stable)
            }
            
            // Keep mist as before
            cloud.setVisible(true)
            cloud.updateAroundHand(joints2D: joints2D,
                                   depthMap: depthMap,
                                   confidenceMap: confMap,
                                   imageResolution: imageRes,
                                   intrinsics: intr)
        } catch {
            // ignore per-frame failures
        }
    }
    
    // Exponential smoothing in 3D
    private func smoothJoints(_ new: [VNHumanHandPoseObservation.JointName: SIMD3<Float>])
    -> [VNHumanHandPoseObservation.JointName: SIMD3<Float>] {
        
        var out: [VNHumanHandPoseObservation.JointName: SIMD3<Float>] = [:]
        out.reserveCapacity(new.count)
        
        for (k, v) in new {
            if let prev = smoothedJoints[k] {
                let blended = prev * smoothing + v * (1 - smoothing)
                out[k] = blended
                smoothedJoints[k] = blended
            } else {
                out[k] = v
                smoothedJoints[k] = v
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

// MARK: - Hand skeleton renderer (bones + simple palm square)
final class HandSkeletonRenderer {
    let root = Entity()
    private var boneEntities: [String: ModelEntity] = [:]
    
    // ✅ Palm quad entity (a flat square for now)
    private var palmEntity: ModelEntity?
    
    private let chains: [[VNHumanHandPoseObservation.JointName]] = [
        [.wrist, .thumbCMC, .thumbMP, .thumbIP, .thumbTip],
        [.wrist, .indexMCP, .indexPIP, .indexDIP, .indexTip],
        [.wrist, .middleMCP, .middlePIP, .middleDIP, .middleTip],
        [.wrist, .ringMCP, .ringPIP, .ringDIP, .ringTip],
        [.wrist, .littleMCP, .littlePIP, .littleDIP, .littleTip]
    ]
    
    private let boneMat = SimpleMaterial(color: .init(white: 0.58, alpha: 0.95),
                                         roughness: 0.35,
                                         isMetallic: false)
    
    // ✅ Slightly different palm material so it reads as a surface
    private let palmMat = SimpleMaterial(color: .init(white: 0.62, alpha: 0.55),
                                         roughness: 0.65,
                                         isMetallic: false)
    
    func setVisible(_ visible: Bool) {
        root.isEnabled = visible
    }
    
    func update(joints3D: [VNHumanHandPoseObservation.JointName: SIMD3<Float>]) {
        // 1) Bones (same as before)
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
        
        // 2) ✅ Palm square (wrist + MCPs)
        updatePalm(joints3D: joints3D)
    }
    
    private func makeBoneCylinder() -> ModelEntity {
        let mesh = MeshResource.generateCylinder(height: 1.0, radius: 0.010)
        return ModelEntity(mesh: mesh, materials: [boneMat])
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
    
    // MARK: - Palm square
    
    /// Builds a simple quad for the palm using 4 joints:
    /// wrist, indexMCP, littleMCP, thumbCMC
    private func updatePalm(joints3D: [VNHumanHandPoseObservation.JointName: SIMD3<Float>]) {
        guard
            let wrist = joints3D[.wrist],
            let indexMCP = joints3D[.indexMCP],
            let littleMCP = joints3D[.littleMCP],
            let thumbCMC = joints3D[.thumbCMC]
        else {
            palmEntity?.isEnabled = false
            return
        }
        
        // Convert to camera-local coords (negative z forward)
        let w = SIMD3<Float>(wrist.x, wrist.y, -wrist.z)
        let i = SIMD3<Float>(indexMCP.x, indexMCP.y, -indexMCP.z)
        let l = SIMD3<Float>(littleMCP.x, littleMCP.y, -littleMCP.z)
        let t = SIMD3<Float>(thumbCMC.x, thumbCMC.y, -thumbCMC.z)
        
        // Quad corners: [thumbCMC, indexMCP, littleMCP, wrist]
        // (Order matters for the face normal)
        let positions: [SIMD3<Float>] = [t, i, l, w]
        
        // If palm doesn't exist yet, create it once
        if palmEntity == nil {
            palmEntity = ModelEntity(mesh: makePalmMesh(positions: positions),
                                     materials: [palmMat])
            if let palmEntity { root.addChild(palmEntity) }
        } else {
            // Update mesh every frame
            palmEntity?.model?.mesh = makePalmMesh(positions: positions)
        }
        
        palmEntity?.isEnabled = true
    }
    
    private func makePalmMesh(positions p: [SIMD3<Float>]) -> MeshResource {
        // 4 vertices, two triangles: (0,1,2) and (0,2,3)
        var desc = MeshDescriptor()
        desc.positions = MeshBuffers.Positions(p)
        desc.primitives = .triangles([0, 1, 2, 0, 2, 3])
        return try! MeshResource.generate(from: [desc])
    }
}

// MARK: - Depth cloud renderer
final class DepthCloudRenderer {
    let root = Entity()
    private var points: [ModelEntity] = []
    private let maxPoints: Int
    
    private let mat = SimpleMaterial(color: .init(white: 0.75, alpha: 0.65),
                                     roughness: 1.0,
                                     isMetallic: false)
    
    init(maxPoints: Int) {
        self.maxPoints = maxPoints
        points.reserveCapacity(maxPoints)
        
        for _ in 0..<maxPoints {
            let e = ModelEntity(mesh: .generateSphere(radius: 0.0028), materials: [mat])
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
                
                let x = (u - cx) / fx * z
                let y = -(v - cy) / fy * z
                
                let e = points[idx]
                e.isEnabled = true
                e.position = SIMD3<Float>(x, y, -z)
                idx += 1
            }
            if idx >= maxPoints { break }
        }
        
        while idx < maxPoints {
            points[idx].isEnabled = false
            idx += 1
        }
    }
}
