; ModuleID = 'check_metal.metal'
source_filename = "check_metal.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.CameraData = type { <3 x float>, <3 x float>, <3 x float>, <3 x float>, %"struct.metal::matrix", %"struct.metal::matrix", <2 x float>, <2 x float> }
%"struct.metal::matrix" = type { [4 x <4 x float>] }
%struct.FrameData = type { <3 x float>, float, float, <3 x i32> }

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define void @testKernel(<3 x i32> noundef %0, %struct.CameraData addrspace(2)* nocapture noundef align 16 dereferenceable(208) "air-buffer-no-alias" %1, %struct.FrameData addrspace(2)* nocapture noundef align 16 dereferenceable(48) "air-buffer-no-alias" %2) local_unnamed_addr #0 {
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!17, !18, !19}
!llvm.ident = !{!20}
!air.version = !{!21}
!air.language_version = !{!22}
!air.source_file_name = !{!23}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{void (<3 x i32>, %struct.CameraData addrspace(2)*, %struct.FrameData addrspace(2)*)* @testKernel, !10, !11}
!10 = !{}
!11 = !{!12, !13, !15}
!12 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid", !"air.arg_unused"}
!13 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 208, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !14, !"air.arg_type_size", i32 208, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"CameraData", !"air.arg_name", !"cam", !"air.arg_unused"}
!14 = !{i32 0, i32 16, i32 0, !"float3", !"position", i32 16, i32 16, i32 0, !"float3", !"forward", i32 32, i32 16, i32 0, !"float3", !"right", i32 48, i32 16, i32 0, !"float3", !"up", i32 64, i32 64, i32 0, !"float4x4", !"unjitteredViewProjection", i32 128, i32 64, i32 0, !"float4x4", !"prevUnjitteredViewProjection", i32 192, i32 8, i32 0, !"float2", !"jitter", i32 200, i32 8, i32 0, !"float2", !"padding"}
!15 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 48, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !16, !"air.arg_type_size", i32 48, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"FrameData", !"air.arg_name", !"frame", !"air.arg_unused"}
!16 = !{i32 0, i32 16, i32 0, !"float3", !"sunDirection", i32 16, i32 4, i32 0, !"float", !"time", i32 20, i32 4, i32 0, !"float", !"deltaTime", i32 32, i32 16, i32 0, !"int3", !"worldOrigin"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{!"Apple metal version 32023.830 (metalfe-32023.830.2)"}
!21 = !{i32 2, i32 8, i32 0}
!22 = !{!"Metal", i32 4, i32 0, i32 0}
!23 = !{!"/Users/rubenvlieger/Documents/RVGRT/check_metal.metal"}
