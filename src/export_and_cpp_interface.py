"""
export_and_cpp_interface.py
===========================
模型导出（ONNX/TensorRT）和C++调用接口规范。

设计原则：
  - Python完成所有训练、验证、ONNX导出
  - C++仅负责：加载ONNX模型 → 推理 → 解析输出张量
  - 后处理（关键点解码、校准片补全、路径规划）在Python中完成，
    导出为独立的ONNX后处理图，或在C++中用简单数学运算复现
"""

from __future__ import annotations
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from model import *


# ══════════════════════════════════════════════════════════════
#  ONNX导出
# ══════════════════════════════════════════════════════════════

def export_to_onnx(
    model_path: str,
    output_path: str,
    image_size: int = 640,
    opset_version: int = 17,
    dynamic_axes: bool = True,
):
    """
    将训练好的PyTorch模型导出为ONNX格式。
    
    动态轴设置允许C++端传入不同尺寸的图像（放大倍率变化时）。
    
    输出节点（ONNX graph outputs）：
      'kp_heatmap'   : float32 (1, 3, H/4, W/4)
      'kp_sigma'     : float32 (1, 3, H/4, W/4)
      'probe_sem'    : float32 (1, 7, H/4, W/4)
      'probe_offset' : float32 (1, 2, H/4, W/4)
      'calib_cls'    : float32 (1, 4, H/4, W/4)
      'calib_reg'    : float32 (1, 5, H/4, W/4)
      'calib_ctr'    : float32 (1, 1, H/4, W/4)
      'scrub_sem'    : float32 (1, 4, H/4, W/4)
      'scrub_offset' : float32 (1, 2, H/4, W/4)
    """

    # 加载模型
    config = {}   # PSEUDOCODE: 从checkpoint恢复config
    model = GSGProbeNet(config)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy_input = torch.zeros(1, 3, image_size, image_size)

    dynamic = None
    if dynamic_axes:
        dynamic = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'kp_heatmap':   {0: 'batch'},
            'probe_sem':    {0: 'batch'},
            'calib_cls':    {0: 'batch'},
            'scrub_sem':    {0: 'batch'},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=[
                'kp_heatmap', 'kp_sigma',
                'probe_sem', 'probe_offset',
                'calib_cls', 'calib_reg', 'calib_ctr',
                'scrub_sem', 'scrub_offset',
            ],
            dynamic_axes=dynamic,
            do_constant_folding=True,
        )
    print(f"ONNX模型已导出: {output_path}")

    # 验证ONNX
    _verify_onnx(output_path, dummy_input.numpy())


def _verify_onnx(onnx_path: str, dummy_input: np.ndarray):
    """PSEUDOCODE: 验证ONNX输出与PyTorch输出的数值一致性"""
    # import onnx, onnxruntime as ort
    # model = onnx.load(onnx_path)
    # onnx.checker.check_model(model)
    # session = ort.InferenceSession(onnx_path)
    # ort_outputs = session.run(None, {'input': dummy_input})
    # assert max_diff < 1e-4
    print("ONNX验证通过（PSEUDOCODE）")


def export_to_tensorrt(
    onnx_path: str,
    trt_path: str,
    fp16: bool = True,
    workspace_gb: int = 4,
):
    """
    将ONNX转换为TensorRT引擎（可选，用于嵌入式/高速推理）。
    PSEUDOCODE: 调用trtexec命令行工具
    命令示例：
      trtexec --onnx=model.onnx --saveEngine=model.trt --fp16 --workspace=4096
    """
    import subprocess
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={trt_path}',
        '--fp16' if fp16 else '',
        f'--workspace={workspace_gb * 1024}',
        '--minShapes=input:1x3x320x320',
        '--optShapes=input:1x3x640x640',
        '--maxShapes=input:1x3x1280x1280',
    ]
    print(f"TensorRT导出命令: {' '.join(cmd)}")
    # subprocess.run(cmd, check=True)


''' c++文件内容 '''
# ══════════════════════════════════════════════════════════════
# §2  C++接口规范（头文件注释形式）
# # ══════════════════════════════════════════════════════════════

CPP_HEADER = '''
// gsg_probe_detector.h
// ====================
// C++ ONNX Runtime调用接口。
// 编译依赖：onnxruntime (>= 1.16), OpenCV (>= 4.5)
// 
// 设计原则：
//   C++只做以下工作：
//   1. 图像预处理（BGR→RGB, resize, normalize）→ 与Python端完全对齐
//   2. ONNX推理（单次forward pass）
//   3. 输出张量的简单后处理（heatmap argmax, sigmoid, NMS）
//   4. 结果填充到 GSGProbeResult 结构体并回传给上层控制系统
//
//   复杂后处理（如校准片遮挡补全、路径规划）在Python中完成，
//   或提前标定为固定参数存入配置文件供C++读取。

#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <array>

namespace gsg_probe {

// ── 结构体定义（与Python definitions.py对齐）─────────────────

struct KeyPoint {
    float x, y;
    float confidence;
    bool visible;
};

struct ProbeKeyPoints {
    KeyPoint G1_inner;   // 左G针右内角
    KeyPoint S_center;   // S针前端中点
    KeyPoint G2_inner;   // 右G针左内角
    float wear_score;
    int   wear_level;    // 0=NORMAL, 1=MILD, 2=SEVERE
};

struct CalibPad {
    int   calib_type;    // 0=LOAD, 1=OPEN, 2=SHORT, 3=THRU
    float center_x, center_y;
    float width, height, angle;
    float confidence;
    float visible_ratio;
};

struct ScrubMark {
    int   probe_label;   // 0=G1, 1=S, 2=G2
    float area_px;
    float centroid_x, centroid_y;
    int   group_id;
};

struct GSGProbeResult {
    bool valid;
    ProbeKeyPoints probe_kps;
    std::vector<CalibPad>   calib_pads;
    std::vector<ScrubMark>  scrub_marks;
    float delta_x_px;
    float delta_y_px;
    float delta_theta_y;
    float inference_time_ms;
};

// ── 检测器类 ──────────────────────────────────────────────────

class GSGProbeDetector {
public:
    /**
     * 构造函数：加载ONNX模型，初始化ONNX Runtime会话。
     * @param model_path  ONNX文件路径
     * @param use_gpu     是否使用GPU (CUDA)
     * @param num_threads CPU线程数（use_gpu=false时有效）
     */
    GSGProbeDetector(
        const std::string& model_path,
        bool use_gpu = false,
        int  num_threads = 4
    );

    ~GSGProbeDetector();

    /**
     * 单帧推理主入口。
     * @param bgr_image  OpenCV BGR图像
     * @param px_per_um  像素物理尺寸比（标定值）
     * @param target_calib_type  目标校准片类型（-1=自动选择）
     * @return GSGProbeResult
     */
    GSGProbeResult detect(
        const cv::Mat& bgr_image,
        float px_per_um = 1.0f,
        int   target_calib_type = -1
    );

    /**
     * 计算探针到目标焊盘的移动量。
     * 输出单位：微米（需除以px_per_um转换）。
     */
    std::array<float, 3> compute_movement(
        const GSGProbeResult& result,
        int target_pad_idx = 0
    );

private:
    // ── 预处理（与Python train transforms对齐）──
    cv::Mat preprocess(const cv::Mat& bgr_image, cv::Size target_size);
    
    // ── 后处理：Heatmap→关键点坐标 ──
    ProbeKeyPoints decode_keypoints(
        const float* heatmap_data,   // shape (3, H/4, W/4)
        const float* sigma_data,
        int hm_h, int hm_w,
        float scale_x, float scale_y
    );

    // ── 后处理：语义图+偏移图→实例掩模 ──
    std::vector<ScrubMark> decode_scrub_marks(
        const float* sem_data,
        const float* offset_data,
        int h, int w
    );

    // ── 后处理：FCOS输出→旋转框 ──
    std::vector<CalibPad> decode_calib_pads(
        const float* cls_data,
        const float* reg_data,
        const float* ctr_data,
        int h, int w,
        float score_thresh = 0.4f
    );

    // 成员变量
    Ort::Env                env_;
    Ort::Session            session_{nullptr};
    Ort::SessionOptions     session_opts_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    int image_size_ = 640;
    float mean_[3] = {0.485f, 0.456f, 0.406f};
    float std_[3]  = {0.229f, 0.224f, 0.225f};
    
    // 几何先验（从Python标定结果读取）
    std::map<int, std::array<float,3>> nominal_pad_sizes_;  // calib_type → {w,h,spacing}
};

} // namespace gsg_probe
'''

CPP_IMPL_SKETCH = '''
// gsg_probe_detector.cpp  (关键实现片段，非完整代码)

#include "gsg_probe_detector.h"
#include <chrono>
#include <algorithm>
#include <numeric>

namespace gsg_probe {

// ── 构造：初始化ONNX Runtime ──────────────────────────────────

GSGProbeDetector::GSGProbeDetector(
    const std::string& model_path, bool use_gpu, int num_threads
) : env_(ORT_LOGGING_LEVEL_WARNING, "gsg_probe") {

    session_opts_.SetIntraOpNumThreads(num_threads);
    session_opts_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    if (use_gpu) {
        // CUDA执行提供者
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = 0;
        session_opts_.AppendExecutionProvider_CUDA(cuda_opts);
    }

    session_ = Ort::Session(env_, model_path.c_str(), session_opts_);

    // 获取输入/输出节点名称
    Ort::AllocatorWithDefaultOptions allocator;
    input_names_  = {"input"};
    output_names_ = {
        "kp_heatmap", "kp_sigma",
        "probe_sem", "probe_offset",
        "calib_cls", "calib_reg", "calib_ctr",
        "scrub_sem", "scrub_offset"
    };
}

// ── 主推理函数 ────────────────────────────────────────────────

GSGProbeResult GSGProbeDetector::detect(
    const cv::Mat& bgr_image, float px_per_um, int target_calib_type
) {
    auto t_start = std::chrono::high_resolution_clock::now();
    GSGProbeResult result;
    result.valid = false;

    // 1. 预处理
    cv::Mat processed = preprocess(bgr_image, cv::Size(image_size_, image_size_));
    
    // 2. BGR→Float32 NCHW张量
    std::vector<float> input_data(1 * 3 * image_size_ * image_size_);
    // mat → NCHW转换（略）
    
    // 3. ONNX推理
    std::vector<int64_t> input_shape = {1, 3, image_size_, image_size_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size()
    );
    
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), output_names_.size()
    );
    
    // 4. 解析各输出
    float scale_x = (float)bgr_image.cols / image_size_;
    float scale_y = (float)bgr_image.rows / image_size_;
    int hm_h = image_size_ / 4, hm_w = image_size_ / 4;
    
    result.probe_kps = decode_keypoints(
        outputs[0].GetTensorData<float>(),   // kp_heatmap
        outputs[1].GetTensorData<float>(),   // kp_sigma
        hm_h, hm_w, scale_x, scale_y
    );
    
    result.scrub_marks = decode_scrub_marks(
        outputs[7].GetTensorData<float>(),   // scrub_sem
        outputs[8].GetTensorData<float>(),   // scrub_offset
        hm_h, hm_w
    );
    
    result.calib_pads = decode_calib_pads(
        outputs[4].GetTensorData<float>(),   // calib_cls
        outputs[5].GetTensorData<float>(),   // calib_reg
        outputs[6].GetTensorData<float>(),   // calib_ctr
        hm_h, hm_w
    );
    
    // 5. 计算控制偏差
    if (!result.calib_pads.empty() && result.probe_kps.S_center.visible) {
        const auto& target = result.calib_pads[0];
        result.delta_x_px = target.center_x - result.probe_kps.S_center.x;
        result.delta_y_px = target.center_y - result.probe_kps.S_center.y;
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    result.inference_time_ms = std::chrono::duration<float, std::milli>(
        t_end - t_start
    ).count();
    result.valid = true;
    return result;
}

// ── Heatmap解码（Argmax + 亚像素精化）────────────────────────

ProbeKeyPoints GSGProbeDetector::decode_keypoints(
    const float* hm, const float* sigma,
    int hm_h, int hm_w, float sx, float sy
) {
    ProbeKeyPoints kps;
    const char* names[3] = {"G1", "S", "G2"};
    
    for (int k = 0; k < 3; ++k) {
        const float* hm_k = hm + k * hm_h * hm_w;
        
        // 找峰值位置
        int peak_idx = std::max_element(hm_k, hm_k + hm_h*hm_w) - hm_k;
        int py = peak_idx / hm_w;
        int px = peak_idx % hm_w;
        float peak_val = hm_k[peak_idx];
        
        // 亚像素精化（3×3 Taylor展开）
        float dx = 0.0f, dy = 0.0f;
        if (px > 0 && px < hm_w-1 && py > 0 && py < hm_h-1) {
            float dxx = hm_k[py*hm_w + px+1] + hm_k[py*hm_w + px-1] - 2*hm_k[py*hm_w+px];
            float dyy = hm_k[(py+1)*hm_w+px] + hm_k[(py-1)*hm_w+px] - 2*hm_k[py*hm_w+px];
            float dxy = (hm_k[(py+1)*hm_w+px+1] - hm_k[(py+1)*hm_w+px-1]
                       - hm_k[(py-1)*hm_w+px+1] + hm_k[(py-1)*hm_w+px-1]) / 4.0f;
            float det = dxx*dyy - dxy*dxy;
            if (std::abs(det) > 1e-6f) {
                dx = -(dyy*(hm_k[py*hm_w+px+1]-hm_k[py*hm_w+px-1])/2
                       - dxy*(hm_k[(py+1)*hm_w+px]-hm_k[(py-1)*hm_w+px])/2) / det;
                dy = -(dxx*(hm_k[(py+1)*hm_w+px]-hm_k[(py-1)*hm_w+px])/2
                       - dxy*(hm_k[py*hm_w+px+1]-hm_k[py*hm_w+px-1])/2) / det;
                dx = std::clamp(dx, -1.0f, 1.0f);
                dy = std::clamp(dy, -1.0f, 1.0f);
            }
        }
        
        // 映射回原图坐标
        float img_x = (px + dx + 0.5f) * 4.0f * sx;
        float img_y = (py + dy + 0.5f) * 4.0f * sy;
        float sigma_val = std::exp(sigma[k*hm_h*hm_w + py*hm_w + px] / 2.0f);
        float conf = std::min(1.0f, peak_val / (sigma_val + 1e-6f));
        
        KeyPoint kp{img_x, img_y, conf, conf > 0.3f};
        if (k == 0) kps.G1_inner = kp;
        else if (k == 1) kps.S_center = kp;
        else kps.G2_inner = kp;
    }
    return kps;
}

} // namespace gsg_probe
'''


# ══════════════════════════════════════════════════════════════
# §3  CMakeLists.txt
# ══════════════════════════════════════════════════════════════

CMAKE_CONTENT = '''
cmake_minimum_required(VERSION 3.18)
project(gsg_probe_detector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 目标库（供上层C++控制系统调用）
add_library(gsg_probe_detector SHARED
    gsg_probe_detector.cpp
)
target_link_libraries(gsg_probe_detector
    onnxruntime
    ${OpenCV_LIBS}
)

# 示例可执行文件（测试用）
add_executable(gsg_probe_test test_detector.cpp)
target_link_libraries(gsg_probe_test gsg_probe_detector)
'''

