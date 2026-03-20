"""
Streamlit app with player annotation and manual event marking support
"""
import streamlit as st
import sys
from pathlib import Path
import cv2
import numpy as np
import tempfile
import time

sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.analyzer import BadmintonAnalyzer
from core.pose_tracker import PoseTracker
import mediapipe as mp

st.set_page_config(
    page_title="羽毛球步法分析 - 标注版",
    page_icon="🏸",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def detect_players_in_frame(frame):
    """检测帧中的所有球员"""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        player_detections = []

        if results.pose_landmarks:
            # 只检测到一个球员（MediaPipe 检测到的主要人物）
            h, w = frame.shape[:2]
            landmarks = results.pose_landmarks.landmark

            # 获取球员边界框
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]

            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)

            player_detections.append({
                'id': 0,
                'bbox': (x_min, y_min, x_max, y_max),
                'landmarks': landmarks
            })

        return player_detections, frame


def main():
    st.markdown('<p class="main-header">🏸 羽毛球步法分析 - 球员标注版</p>', unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 分析设置")

        # 分析模式
        analysis_mode = st.radio(
            "分析模式",
            ["👤 单人自动分析", "👥 多人选择分析", "📝 手动标注分析"],
            index=0
        )

        st.divider()

        # 参数设置
        model_complexity = st.slider(
            "模型精度",
            min_value=0,
            max_value=2,
            value=1,
            help="0=快速, 1=平衡, 2=精确"
        )

        reference_level = st.selectbox(
            "参考水平",
            ["professional", "advanced", "intermediate"],
            format_func=lambda x: {
                "professional": "🏆 专业选手",
                "advanced": "⭐ 高级业余",
                "intermediate": "📈 中级水平"
            }.get(x, x)
        )

    # 主内容区域
    if analysis_mode == "👤 单人自动分析":
        single_player_analysis(reference_level, model_complexity)
    elif analysis_mode == "👥 多人选择分析":
        multi_player_selection(reference_level, model_complexity)
    else:
        manual_annotation_analysis(reference_level, model_complexity)


def single_player_analysis(reference_level, model_complexity):
    """单人自动分析模式"""
    st.subheader("👤 单人自动分析")

    uploaded_file = st.file_uploader(
        "上传羽毛球比赛视频",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="视频应包含清晰的球员全身"
    )

    if uploaded_file is not None:
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # 预览视频
        col1, col2 = st.columns([2, 1])

        with col1:
            st.video(video_path)

        with col2:
            st.subheader("分析选项")

            generate_annotated = st.checkbox("生成标注视频", value=True)
            show_heatmap = st.checkbox("生成热力图", value=True)

            if st.button("🚀 开始分析", type="primary"):
                with st.spinner("正在分析视频..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def progress_callback(frame_idx, total_frames):
                        progress = min(1.0, frame_idx / max(1, total_frames - 1))
                        progress_bar.progress(progress)
                        status_text.text(f"处理中... {frame_idx}/{total_frames} 帧")

                    try:
                        output_dir = Path(__file__).parent.parent / "data" / "output"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = str(output_dir / f"analyzed_{int(time.time())}.mp4") if generate_annotated else None

                        with BadmintonAnalyzer(
                            model_complexity=model_complexity,
                            reference_level=reference_level,
                        ) as analyzer:
                            result = analyzer.process_video(
                                video_path=video_path,
                                output_path=output_path,
                                progress_callback=progress_callback,
                            )

                        progress_bar.empty()
                        status_text.empty()

                        st.success("✅ 分析完成！")
                        display_results(result)

                    except Exception as e:
                        st.error(f"分析失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())


def multi_player_selection(reference_level, model_complexity):
    """多人选择分析模式"""
    st.subheader("👥 多人选择分析")

    uploaded_file = st.file_uploader(
        "上传包含多个球员的视频",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="视频应包含2个或更多球员"
    )

    if uploaded_file is not None:
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        st.info("👆 点击下方按钮检测视频中的球员")

        if st.button("🔍 检测球员"):
            with st.spinner("正在检测球员..."):
                # 读取视频第一帧
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    # 检测球员
                    player_detections, annotated_frame = detect_players_in_frame(frame)

                    if player_detections:
                        st.success(f"检测到 {len(player_detections)} 个球员")

                        # 显示检测结果
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("检测结果")
                            # 转换为RGB显示
                            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, use_container_width=True)

                        with col2:
                            st.subheader("选择要分析的球员")
                            for i, player in enumerate(player_detections):
                                if st.button(f"球员 {i+1}", key=f"player_{i}"):
                                    st.session_state.selected_player_id = i
                                    st.success(f"已选择球员 {i+1}")

                            if st.session_state.get('selected_player_id') is not None:
                                st.info(f"当前选择: 球员 {st.session_state.selected_player_id + 1}")

                                if st.button("🚀 开始分析选中球员", type="primary"):
                                    # TODO: 实现针对特定球员的分析
                                    st.info("功能开发中...")
                    else:
                        st.warning("未检测到球员，请确保视频中有人物")


def manual_annotation_analysis(reference_level, model_complexity):
    """手动标注分析模式"""
    st.subheader("📝 手动标注分析")

    uploaded_file = st.file_uploader(
        "上传视频",
        type=['mp4', 'avi', 'mov', 'mkv']
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # 视频浏览和标注界面
        st.info("🎯 使用下方��件浏览视频，在关键时刻点击'标记事件'")

        # 创建两列布局
        col1, col2 = st.columns([3, 1])

        with col1:
            # 视频播放器占位符
            video_placeholder = st.empty()

        with col2:
            st.subheader("标注工具")

            # 帧控制
            frame_slider = st.slider("帧位置", 0, 100, 0, key="frame_slider")

            # 事件类型
            event_type = st.selectbox(
                "事件类型",
                ["击球", "得分", "失误", "跳杀", "网前球", "其他"]
            )

            # 标注按钮
            if st.button("📍 标记当前事件", type="primary"):
                if 'annotations' not in st.session_state:
                    st.session_state.annotations = []

                st.session_state.annotations.append({
                    'frame': frame_slider,
                    'type': event_type,
                    'timestamp': frame_slider / 30.0  # 假设30fps
                })

                st.success(f"已标记: {event_type} @ 帧 {frame_slider}")

            # 显示已标注的事件
            st.divider()
            st.subheader("已标注事件")

            if 'annotations' in st.session_state and st.session_state.annotations:
                for i, ann in enumerate(st.session_state.annotations):
                    st.write(f"{i+1}. {ann['type']} - 帧 {ann['frame']} ({ann['timestamp']:.2f}s)")
            else:
                st.info("暂无标注")

            # 导出按钮
            if st.button("💾 导出标注"):
                if 'annotations' in st.session_state:
                    import json
                    import pandas as pd

                    df = pd.DataFrame(st.session_state.annotations)
                    csv = df.to_csv(index=False)

                    st.download_button(
                        label="📥 下载标注文件 (CSV)",
                        data=csv,
                        file_name="annotations.csv",
                        mime="text/csv"
                    )

        # 显示视频
        st.video(video_path)


def display_results(result):
    """显示分析结果"""
    from core.analyzer import AnalysisResult

    st.divider()
    st.header("📊 分析结果")

    # 综合评分
    score = result.efficiency_score

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("综合评分", f"{score.overall:.0f}/100")
    with col2:
        st.metric("移动效率", f"{score.movement_efficiency:.0f}")
    with col3:
        st.metric("反应速度", f"{score.response_time:.0f}")
    with col4:
        st.metric("场地覆盖", f"{score.court_coverage:.0f}")
    with col5:
        st.metric("平衡稳定", f"{score.balance_stability:.0f}")

    # 详细指标
    st.subheader("📈 详细指标")

    metrics = result.metrics

    tab1, tab2, tab3 = st.tabs(["基础数据", "效率对比", "建议"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**移动数据**")
            st.json({
                "总步数": metrics.total_steps,
                "步频": f"{metrics.step_frequency:.2f} 步/秒",
                "平均步长": f"{metrics.avg_step_length:.3f} m",
                "总距离": f"{metrics.total_distance:.2f} m",
            })

        with col2:
            st.markdown("**事件统计**")
            st.json({
                "跳跃次数": metrics.jump_count,
                "变向次数": metrics.direction_changes,
                "路径效率": f"{metrics.path_efficiency:.2%}",
            })

    with tab2:
        st.markdown("**与专业选手对比**")
        for metric_name, comp in result.comparisons.items():
            st.write(f"**{metric_name}**: {comp.assessment}")
            st.progress(min(1.0, abs(comp.difference) / max(0.01, abs(comp.reference_value))))

    with tab3:
        if result.recommendations:
            for i, rec in enumerate(result.recommendations[:3], 1):
                st.info(f"**{i}. {rec['area']}** [{rec['priority'].upper()}]\n\n{rec['recommendation']}")


if __name__ == "__main__":
    main()
