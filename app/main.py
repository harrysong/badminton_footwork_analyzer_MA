"""
Streamlit main application for Badminton Footwork Analyzer
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import tempfile
import time

from core.analyzer import BadmintonAnalyzer, AnalysisResult
from visualization.trajectory_plotter import TrajectoryPlotter

# Page config
st.set_page_config(
    page_title="羽毛球步法效率分析系统",
    page_icon="🏸",
    layout="wide",
    initial_sidebar_state="expanded",
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
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .recommendation-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'components_loaded' not in st.session_state:
    # Components already imported at top, just mark as loaded
    st.session_state.components_loaded = True
if 'custom_metrics' not in st.session_state:
    # Default custom metrics to analyze
    st.session_state.custom_metrics = {
        'recovery_speed': True,      # 回中速度
        'net_approach': True,        # 上网速度
        'backward_speed': True,      # 后退速度
        'lateral_speed': True,       # 左右移动速度
        'crouch_depth': True,        # 身体下蹲
        'split_step_timing': True,   # 分腿跳时机
        'first_step_speed': True,    # 第一步速度
    }


def main():
    st.markdown('<p class="main-header">🏸 羽毛球步法效率分析系统</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于计算机视觉的步法分析与效率评估</p>', unsafe_allow_html=True)

    # Component loading status
    if not st.session_state.get('components_loaded', False):
        error = st.session_state.get('load_error', 'Unknown error')
        st.error(f"❌ 组件加载失败: {error}")
        st.stop()
    else:
        # Show success indicator (small and subtle)
        st.markdown(
            """
            <div style='text-align: right; padding: 5px;'>
                <span style='color: #28a745; font-size: 0.8rem;'>● 系统就绪</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ 设置")

        analysis_mode = st.radio(
            "分析模式",
            ["📹 视频分析", "🎯 对手分析", "👥 球员选择分析"],
            index=0
        )

        st.divider()

        # Analysis settings
        st.subheader("分析参数")
        reference_level = st.selectbox(
            "参考水平",
            ["professional", "advanced", "intermediate"],
            index=0,
            format_func=lambda x: {
                "professional": "🏆 专业选手",
                "advanced": "⭐ 高级业余",
                "intermediate": "📈 中级水平"
            }.get(x, x)
        )

        model_complexity = st.slider(
            "模型精度",
            min_value=0,
            max_value=2,
            value=1,
            help="0=快速, 1=平衡, 2=精确(慢，可���不稳定)"
        )

        if model_complexity == 2:
            st.warning("⚠️ 模型精度 2 在某些系统上可能不稳定，建议使用精度 1")

        st.divider()

        # Custom metrics selection
        st.subheader("🎯 自定义分析指标")

        custom_metrics = st.session_state.custom_metrics

        st.markdown("**基础移动分析**")
        custom_metrics['recovery_speed'] = st.checkbox(
            "⬅️ 回中速度", value=True, key='metric_recovery',
            help="分析回到场地中心的速度"
        )
        custom_metrics['net_approach'] = st.checkbox(
            "⬆️ 上网速度", value=True, key='metric_net',
            help="分析向网前移动的速度"
        )
        custom_metrics['backward_speed'] = st.checkbox(
            "⬇️ 后退速度", value=True, key='metric_backward',
            help="分析向后场移动的速度"
        )
        custom_metrics['lateral_speed'] = st.checkbox(
            "↔️ 左右移动速度", value=True, key='metric_lateral',
            help="分析横向移动速度"
        )

        st.markdown("**技术细节分析**")
        custom_metrics['crouch_depth'] = st.checkbox(
            "🦵 身体下蹲深度", value=True, key='metric_crouch',
            help="分析接球时的身体下蹲程度"
        )
        custom_metrics['split_step_timing'] = st.checkbox(
            "🦘 分腿跳时机", value=True, key='metric_split',
            help="分析分腿跳的时机准确性"
        )
        custom_metrics['first_step_speed'] = st.checkbox(
            "🚀 第一步速度", value=True, key='metric_first',
            help="分析启动后的第一步速度"
        )

        # Update session state
        st.session_state.custom_metrics = custom_metrics

        st.divider()

        # Detection quality tips
        st.subheader("💡 提高检测率")
        with st.expander("查看如何提高运动员识别率"):
            st.markdown("""
            **如果检测不到运动员，请检查**:

            **📹 视频质量**
            - ✅ 光线充足，避免过暗或过度曝光
            - ✅ 画面清晰，避免运动模糊
            - ✅ 分辨率至少 720p，推荐 1080p
            - ❌ 避免镜头抖动

            **📐 拍摄角度**
            - ✅ 从球场���面拍摄（能看到全身）
            - ✅ 从正面/背面拍摄
            - ❌ 避免俯视或仰视角度

            **🏸 场景要求**
            - ✅ 运动员全身在画面中
            - ✅ 避免球网或其他物体遮挡
            - ✅ 避免多人重叠
            - ✅ 背景尽量简单整洁

            **⚙️ 模型设置**
            - 检测率低 → 使用模型精度 **0** (最快)
            - 追求精度 → 使用模型精度 **1** (推荐)
            - ❌ 避免使用精度 2 (可能不稳定)

            **🔍 常见问题**
            - 运动员太小 → 放大拍摄
            - 运动太快 → 提高视频帧率
            - 服装颜色与背景相似 → 增加对比度
            """)

    # Main content
    if analysis_mode == "📹 视频分析":
        video_analysis_page(reference_level, model_complexity)
    elif analysis_mode == "🎯 对手分析":
        opponent_analysis_page(reference_level, model_complexity)
    else:
        player_selection_page(reference_level, model_complexity)

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #888; font-size: 0.85rem;'>
            💡 提示：上传视频后点击"开始分析"按钮即可生成步法分析报告
        </div>
        """,
        unsafe_allow_html=True
    )


def video_analysis_page(reference_level: str, model_complexity: int):
    """Video analysis page"""

    # Welcome section
    st.markdown("### 📹 视频分析")
    st.markdown("""
    本系统可以分析您的羽毛球步法效率，包括：
    - 🏃 **移动效率分析** - 路径效率、步频、步幅
    - ⚡ **反应时间评估** - 启动响应时间
    - 🗺️ **场地覆盖分析** - 活动热力图
    - 📊 **专业水平对��** - 与专业选手数据对比
    """)

    st.divider()

    # Welcome message when no file is uploaded
    st.info("👆 请在下方上传一个羽毛球比赛视频开始分析")

    # File upload
    uploaded_file = st.file_uploader(
        "上传羽毛球比赛视频",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="支持 MP4, AVI, MOV, MKV 格式",
        label_visibility="visible"
    )

    # Reset state when new file is uploaded
    if uploaded_file is not None:
        current_file_name = uploaded_file.name
        if st.session_state.get('video_last_uploaded_file') != current_file_name:
            st.session_state.result = None
            st.session_state.video_last_uploaded_file = current_file_name

    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # Display video
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📹 原始视频")
            st.video(video_path)

        with col2:
            st.subheader("🎛️ 分析选项")

            generate_annotated = st.checkbox(
                "生成标注视频",
                value=True,
                help="生成带有姿态估计和轨迹标注的输出视频"
            )

            show_heatmap = st.checkbox(
                "生成热力图",
                value=True,
                help="显示步法活动热力图"
            )

            st.divider()

            if st.button("🚀 开始分析", type="primary", use_container_width=True):
                process_video(video_path, reference_level, model_complexity, generate_annotated)

    # Display results
    if st.session_state.result is not None:
        display_results(st.session_state.result, show_heatmap)


def process_video(video_path: str, reference_level: str, model_complexity: int, generate_annotated: bool):
    """Process video with progress bar"""
    st.session_state.is_processing = True

    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(frame_idx: int, total_frames: int):
        progress = min(1.0, frame_idx / max(1, total_frames - 1))
        progress_bar.progress(progress)
        status_text.text(f"处理中... {frame_idx}/{total_frames} 帧 ({progress*100:.1f}%)")

    # Create analyzer
    with BadmintonAnalyzer(
        model_complexity=model_complexity,
        reference_level=reference_level,
    ) as analyzer:

        output_path = None
        if generate_annotated:
            # Create output directory in project
            output_dir = Path(__file__).parent.parent / "data" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"analyzed_{int(time.time())}.mp4")

        status_text.text("正在分析视频，请稍候...")

        # Process video
        result = analyzer.process_video(
            video_path=video_path,
            output_path=output_path,
            progress_callback=progress_callback,
        )

        st.session_state.result = result

    progress_bar.empty()
    status_text.empty()
    st.session_state.is_processing = False

    st.success("✅ 分析完成！")

    # Display detection statistics to help users understand detection quality
    if hasattr(analyzer, 'pose_tracker'):
        stats = analyzer.pose_tracker.get_detection_stats()
        detection_rate = stats['detection_rate'] * 100

        # Only show warnings if detection rate is concerning
        if detection_rate < 50:
            st.error(f"""
            ⚠️ **检测率较低**: {detection_rate:.1f}%
            - 检测到: {stats['detected_frames']} 帧
            - 未检测到: {stats['missing_frames']} 帧
            - 平均置信度: {stats['last_confidence']:.2f}

            **这可能的原因**:
            1. 运动员被部分遮挡（被球网、其他球员）
            2. 光线太暗或过度曝光
            3. 运动模糊导致关键点不清晰
            4. 运动员在画面外或只显示部分身体
            5. 拍摄角度不佳（建议从侧面或正面拍摄）

            **建议**:
            - 使用光线充足的视频
            - 确保运动员全身在画面中
            - 避免球网或其他遮挡物
            - 选择模型精度 0 以获得更高的检测速度
            """)
        elif detection_rate < 80:
            st.warning(f"""
            📊 **检测率**: {detection_rate:.1f}% ({stats['detected_frames']}/{stats['total_frames']} 帧检测到姿态)
            平均置信度: {stats['last_confidence']:.2f}
            """)


def display_advanced_metrics(result):
    """Display advanced biomechanics and tactical analysis"""
    st.markdown("### 🔬 高级分析")

    # Check if advanced metrics are available
    if not hasattr(result, 'biomechanics') or not hasattr(result, 'tactical_geometry'):
        st.info("💡 高级指标正在计算中，需要更多击球数据...")
        st.caption("提示：请确保视频中包含清晰的击球动作以获得准确分析")
        return

    biomech = result.biomechanics
    tactical = result.tactical_geometry
    rhythm = result.rhythm_control

    # Show shot count
    if hasattr(result, 'shots') and result.shots:
        st.info(f"✅ 检测到 {len(result.shots)} 次击球动作")
    else:
        st.warning("⚠️ 未检测到足够的击球动作，高级指标可能不准确")
        return

    # Create sub-tabs for different analysis categories
    subtab1, subtab2, subtab3 = st.tabs(["🦾 生物力学", "🎯 战术几何", "🎵 节奏控制"])

    with subtab1:
        st.markdown("#### 🦾 击球生物力学分析")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**动力链同步性**")

            sync_score = biomech.kinetic_chain_synchronization * 100
            sync_status = "优秀" if sync_score > 80 else "良好" if sync_score > 60 else "需改进"

            st.metric(
                "同步性评分",
                f"{sync_score:.0f}%",
                delta=sync_status
            )

            st.progress(sync_score / 100)

            if sync_score < 60:
                st.warning("""
                💡 **改进建议**：
                - 练习全身协调发力
                - 加强转髋和肩部旋转的配合
                - 使用慢动作挥拍练习力量传递顺序
                """)

        with col2:
            st.markdown("**击球点分析**")

            st.metric("最高击球点", f"{biomech.highest_hit_point:.2f}m")
            st.metric("平均击球高度", f"{biomech.avg_hit_point_height:.2f}m")
            st.metric("一致性", f"±{biomech.hit_point_consistency:.3f}m")

            if biomech.highest_hit_point < 2.5:
                st.warning("""
                💡 **改进建议**：
                - 击球点偏低，建议加强跳跃扣杀练习
                - 提高起跳时机和高度
                - 练习在最高点击球
                """)

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**挥拍速度**")
            st.metric("平均角速度", f"{biomech.avg_swing_angular_velocity:.1f} rad/s")
            st.metric("最大角速度", f"{biomech.max_swing_angular_velocity:.1f} rad/s")

        with col4:
            st.markdown("**力量传递**")
            power_pct = biomech.power_transfer_efficiency * 100
            st.metric("传递效率", f"{power_pct:.0f}%")
            st.progress(power_pct / 100)

    with subtab2:
        st.markdown("#### 🎯 战术几何与空间分析")

        # 16-grid distribution visualization
        if tactical.shot_distribution_16grid:
            st.markdown("**击球分布热力图（16宫格）**")

            # Convert to dataframe for display
            grid_data = []
            for zone, count in sorted(tactical.shot_distribution_16grid.items()):
                row, col = int(zone[0]), int(zone[1])
                grid_data.append({
                    "区域": f"({row},{col})",
                    "次数": count,
                    "占比": f"{count/sum(tactical.shot_distribution_16grid.values())*100:.1f}%"
                })

            if grid_data:
                df_grid = pd.DataFrame(grid_data)
                st.dataframe(df_grid, use_container_width=True)

                # Preferred zones
                if tactical.preferred_zones:
                    st.markdown("**偏好区域** (最常击球位置)")
                    for zone in tactical.preferred_zones[:5]:
                        count = tactical.shot_distribution_16grid.get(zone, 0)
                        st.write(f"- 区域 **{zone}**: {count} 次")

        st.markdown("---")

        col5, col6 = st.columns(2)

        with col5:
            st.markdown("**线路准确性**")
            accuracy = tactical.optimal_line_accuracy * 100
            st.metric("几何最优匹配", f"{accuracy:.0f}%")
            st.caption("反映实际出球与理论最优线路的匹配度")

        with col6:
            st.markdown("**空档暴露**")
            st.metric("平均暴露面积", f"{tactical.average_exposed_area:.1f} m²")
            st.metric("最大暴露面积", f"{tactical.max_exposed_area:.1f} m²")
            st.caption("击球后留给对手可利用的空间")

    with subtab3:
        st.markdown("#### 🎵 节奏控制分析")

        if rhythm.inter_shot_intervals:
            st.markdown("**多拍节奏分布**")

            intervals_df = pd.DataFrame({
                "间隔(秒)": rhythm.inter_shot_intervals,
                "拍序": range(1, len(rhythm.inter_shot_intervals) + 1)
            })

            # Plot rhythm
            fig = px.line(
                intervals_df,
                x="拍序",
                y="间隔(秒)",
                title="拍间时间间隔分布",
                markers=True
            )
            fig.add_hline(
                y=rhythm.avg_rally_tempo,
                line_dash="dash",
                annotation_text=f"平均: {rhythm.avg_rally_tempo:.2f}s"
            )
            st.plotly_chart(fig, use_container_width=True)

        col7, col8, col9 = st.columns(3)

        with col7:
            st.metric("平均拉锯节奏", f"{rhythm.avg_rally_tempo:.2f}s")
            st.metric("节奏变化", f"±{rhythm.rhythm_variance:.2f}s")

        with col8:
            st.metric("平均回合长度", f"{rhythm.avg_rally_length:.1f}拍")

        with col9:
            attack_pct = rhythm.attack_frequency * 100
            st.metric("进攻频率", f"{attack_pct:.0f}%")

        st.markdown("**挥拍动作一致性**")
        consistency_pct = rhythm.swing_consistency_score * 100
        st.metric("一致性评分", f"{consistency_pct:.0f}%")
        st.progress(consistency_pct / 100)
        st.caption("同一击球点的挥拍动作重复性")


def display_custom_metrics(metrics):
    """Display custom metrics based on user selection"""
    custom_metrics = st.session_state.get('custom_metrics', {})

    st.markdown("### 🎯 自定义指标分析")

    if not any(custom_metrics.values()):
        st.info("💡 请在左侧边栏选择要分析的指标")
        return

    # Collect all selected metrics
    selected_metrics = {k: v for k, v in custom_metrics.items() if v}

    # Create metric display data
    metric_sections = []

    # Basic Movement Metrics
    basic_movement = []

    if selected_metrics.get('recovery_speed'):
        basic_movement.append({
            "指标": "⬅️ 回中速度",
            "数值": f"{metrics.recovery_speed:.2f} m/s" if metrics.recovery_speed > 0 else "N/A",
            "事件数": metrics.recovery_events,
        })

    if selected_metrics.get('net_approach'):
        basic_movement.append({
            "指标": "⬆️ 上网速度",
            "数值": f"{metrics.net_approach_speed:.2f} m/s" if metrics.net_approach_speed > 0 else "N/A",
            "事件数": metrics.net_approach_events,
        })

    if selected_metrics.get('backward_speed'):
        basic_movement.append({
            "指标": "⬇️ 后退速度",
            "数值": f"{metrics.backward_speed:.2f} m/s" if metrics.backward_speed > 0 else "N/A",
            "事件数": metrics.backward_events,
        })

    if selected_metrics.get('lateral_speed'):
        basic_movement.append({
            "指标": "↔️ 左右移动速度",
            "数值": f"{metrics.lateral_speed:.2f} m/s" if metrics.lateral_speed > 0 else "N/A",
            "左移": f"{metrics.lateral_left_speed:.2f} m/s" if metrics.lateral_left_speed > 0 else "N/A",
            "右移": f"{metrics.lateral_right_speed:.2f} m/s" if metrics.lateral_right_speed > 0 else "N/A",
        })

    if basic_movement:
        st.markdown("#### 🏃 基础移动指标")
        col1, col2 = st.columns(2)

        with col1:
            for m in basic_movement[:2]:
                st.markdown(f"**{m['指标']}**")
                st.metric("速度", m['数值'])

        with col2:
            for m in basic_movement[2:]:
                st.markdown(f"**{m['指标']}**")
                st.metric("速度", m['数值'])

        # Detailed table
        df_basic = pd.DataFrame(basic_movement)
        st.dataframe(df_basic, use_container_width=True)

    # Technical Detail Metrics
    tech_detail = []

    if selected_metrics.get('crouch_depth'):
        crouch_status = "深蹲" if metrics.crouch_depth_avg < 0.3 else "中等" if metrics.crouch_depth_avg < 0.5 else "高姿态"
        tech_detail.append({
            "指标": "🦵 身体下蹲深度",
            "平均值": f"{metrics.crouch_depth_avg:.2f}",
            "最低值": f"{metrics.crouch_depth_min:.2f}",
            "状态": crouch_status,
            "采样数": metrics.crouch_events,
        })

    if selected_metrics.get('split_step_timing'):
        timing_pct = metrics.split_step_timing_accuracy * 100
        timing_status = "优秀" if timing_pct > 80 else "良好" if timing_pct > 60 else "需改进"
        tech_detail.append({
            "指标": "🦘 分腿跳时机",
            "准确性": f"{timing_pct:.1f}%",
            "评估": timing_status,
            "正确次数": f"{metrics.split_step_before_movement}/{max(1, metrics.jump_count)}",
        })

    if selected_metrics.get('first_step_speed'):
        tech_detail.append({
            "指标": "🚀 第一步速度",
            "速度": f"{metrics.first_step_speed:.2f} m/s" if metrics.first_step_speed > 0 else "N/A",
            "响应时间": f"{metrics.first_step_time:.3f} s" if metrics.first_step_time > 0 else "N/A",
        })

    if tech_detail:
        st.markdown("#### 🎯 技术细节指标")
        df_tech = pd.DataFrame(tech_detail)
        st.dataframe(df_tech, use_container_width=True)

    # Speed Comparison Chart
    if any([
        selected_metrics.get('recovery_speed') and metrics.recovery_speed > 0,
        selected_metrics.get('net_approach') and metrics.net_approach_speed > 0,
        selected_metrics.get('backward_speed') and metrics.backward_speed > 0,
        selected_metrics.get('lateral_speed') and metrics.lateral_speed > 0,
        selected_metrics.get('first_step_speed') and metrics.first_step_speed > 0,
    ]):
        st.markdown("#### 📊 速度对比图")

        speed_labels = []
        speed_values = []

        if selected_metrics.get('recovery_speed') and metrics.recovery_speed > 0:
            speed_labels.append("回中速度")
            speed_values.append(metrics.recovery_speed)
        if selected_metrics.get('net_approach') and metrics.net_approach_speed > 0:
            speed_labels.append("上网速度")
            speed_values.append(metrics.net_approach_speed)
        if selected_metrics.get('backward_speed') and metrics.backward_speed > 0:
            speed_labels.append("后退速度")
            speed_values.append(metrics.backward_speed)
        if selected_metrics.get('lateral_speed') and metrics.lateral_speed > 0:
            speed_labels.append("横向速度")
            speed_values.append(metrics.lateral_speed)
        if selected_metrics.get('first_step_speed') and metrics.first_step_speed > 0:
            speed_labels.append("第一步速度")
            speed_values.append(metrics.first_step_speed)

        if speed_labels:
            fig = go.Figure(data=[
                go.Bar(
                    x=speed_labels,
                    y=speed_values,
                    marker=dict(
                        color=speed_values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="速度 (m/s)")
                    ),
                    text=[f"{v:.2f} m/s" for v in speed_values],
                    textposition='auto',
                )
            ])

            fig.update_layout(
                title="各方向移动速度对比",
                xaxis_title="移动类型",
                yaxis_title="速度 (m/s)",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

    # Recommendations based on custom metrics
    if selected_metrics:
        st.markdown("#### 💡 基于自定义指标的建议")

        custom_recommendations = []

        if selected_metrics.get('recovery_speed') and metrics.recovery_speed > 0:
            if metrics.recovery_speed < 1.5:
                custom_recommendations.append({
                    "area": "回中速度",
                    "issue": f"回中速度({metrics.recovery_speed:.2f}m/s)较慢",
                    "recommendation": "建议加强击球后立即回位的意识训练。练习'击球-回位'的连贯动作，减少在场地边缘的停留时间。",
                    "priority": "medium",
                })

        if selected_metrics.get('net_approach') and metrics.net_approach_speed > 0:
            if metrics.net_approach_speed < 2.0:
                custom_recommendations.append({
                    "area": "上网速度",
                    "issue": f"上网速度({metrics.net_approach_speed:.2f}m/s)偏慢",
                    "recommendation": "建议加强向前跨步训练，重点练习蹬跨步和交叉步上网。提高下肢爆发力。",
                    "priority": "high",
                })

        if selected_metrics.get('backward_speed') and metrics.backward_speed > 0:
            if metrics.backward_speed < 1.8:
                custom_recommendations.append({
                    "area": "后退速度",
                    "issue": f"后退速度({metrics.backward_speed:.2f}m/s)偏慢",
                    "recommendation": "建议加强后退步法训练，练习并步后退和交叉步后退。注意保持身体平衡，避免后退时失去重心。",
                    "priority": "high",
                })

        if selected_metrics.get('lateral_speed') and metrics.lateral_speed > 0:
            if metrics.lateral_speed < 2.0:
                custom_recommendations.append({
                    "area": "左右移动",
                    "issue": f"横向移动速度({metrics.lateral_speed:.2f}m/s)偏慢",
                    "recommendation": "建议加强横向步法训练，练习侧并步和交叉步。重点提高髋关节灵活性和侧向移动能力。",
                    "priority": "medium",
                })

        if selected_metrics.get('crouch_depth') and metrics.crouch_depth_avg > 0:
            if metrics.crouch_depth_avg > 0.5:
                custom_recommendations.append({
                    "area": "身体重心",
                    "issue": f"平均下蹲深度({metrics.crouch_depth_avg:.2f})偏高，重心偏上",
                    "recommendation": "建议在准备姿势和移动时保持更低重心。练习膝盖弯曲、身体前倾的姿势，这有助于快速启动和稳定击球。",
                    "priority": "medium",
                })

        if selected_metrics.get('split_step_timing') and metrics.split_step_timing_accuracy > 0:
            if metrics.split_step_timing_accuracy < 0.6:
                custom_recommendations.append({
                    "area": "分腿跳时机",
                    "issue": f"分腿跳时机准确性({metrics.split_step_timing_accuracy*100:.0f}%)较低",
                    "recommendation": "建议在对手击球瞬间完成分腿跳。练习观察对手挥拍动作，提前预判击球方向。使用多球训练强化时机感。",
                    "priority": "high",
                })

        if selected_metrics.get('first_step_speed') and metrics.first_step_speed > 0:
            if metrics.first_step_speed < 2.5:
                custom_recommendations.append({
                    "area": "第一步启动",
                    "issue": f"第一步速度({metrics.first_step_speed:.2f}m/s)较慢",
                    "recommendation": "第一步启动速度直接影响到位及时性。建议加强反应训练和爆发力练习。练习听口令或看手势快速启动。",
                    "priority": "high",
                })

        # Display recommendations
        if custom_recommendations:
            priority_order = {"high": 0, "medium": 1, "low": 2}
            custom_recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))

            for i, rec in enumerate(custom_recommendations[:5], 1):
                priority = rec.get('priority', 'low')
                color = "#ea580c" if priority == "high" else "#ca8a04"

                st.markdown(f"""
                <div style="border-left: 4px solid {color}; padding: 12px; margin-bottom: 10px; background-color: #fafafa; border-radius: 6px;">
                    <div style="font-weight: bold; margin-bottom: 4px;">{i}. {rec['area']}</div>
                    <div style="color: #666; margin-bottom: 8px;">问题：{rec['issue']}</div>
                    <div style="background-color: #f0f7ff; padding: 8px; border-radius: 4px;">{rec['recommendation']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("✅ 当前选择的指标表现良好！")


def display_results(result: AnalysisResult, show_heatmap: bool):
    """Display analysis results"""

    st.divider()
    st.header("📊 分析结果")

    # Overall score
    score = result.efficiency_score

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "综合评分",
            f"{score.overall:.0f}/100",
            delta=None
        )

    with col2:
        st.metric(
            "移动效率",
            f"{score.movement_efficiency:.0f}"
        )

    with col3:
        st.metric(
            "反应速度",
            f"{score.response_time:.0f}"
        )

    with col4:
        st.metric(
            "场地覆盖",
            f"{score.court_coverage:.0f}"
        )

    with col5:
        st.metric(
            "平衡稳定",
            f"{score.balance_stability:.0f}"
        )

    # Detailed metrics
    st.subheader("📈 详细指标")

    metrics = result.metrics

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["基础数据", "效率对比", "可视化", "🎯 自定义指标", "🔬 高级分析"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**移动数据**")
            basic_data = {
                "总步数": metrics.total_steps,
                "步频": f"{metrics.step_frequency:.2f} 步/秒",
                "平均步长": f"{metrics.avg_step_length:.3f} m",
                "最大步长": f"{metrics.max_step_length:.3f} m",
                "总距离": f"{metrics.total_distance:.2f} m",
                "平均速度": f"{metrics.avg_speed:.2f} m/s",
                "最大速度": f"{metrics.max_speed:.2f} m/s",
            }
            st.json(basic_data)

        with col2:
            st.markdown("**事件统计**")
            events_data = {
                "跳跃次数": metrics.jump_count,
                "变向次数": metrics.direction_changes,
                "路径效率": f"{metrics.path_efficiency:.2%}",
                "场地覆盖率": f"{metrics.coverage_ratio:.2%}",
                "平均反应时间": f"{metrics.avg_response_time:.3f} s" if metrics.avg_response_time > 0 else "N/A",
                "CoM稳定性": f"{metrics.com_stability:.4f}",
            }
            st.json(events_data)

    with tab2:
        st.markdown("**与专业选手对比**")

        comparison_data = []
        for metric_name, comp in result.comparisons.items():
            comparison_data.append({
                "指标": metric_name,
                "你的数据": f"{comp.player_value:.3f}",
                "参考值": f"{comp.reference_value:.3f}",
                "差距": f"{comp.difference:+.3f}",
                "评估": comp.assessment,
            })

        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)

        # Radar chart
        categories = list(result.comparisons.keys())[:5]
        player_values = [result.comparisons[c].player_value for c in categories]
        reference_values = [result.comparisons[c].reference_value for c in categories]

        # Normalize values for radar chart
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=player_values,
            theta=categories,
            fill='toself',
            name='你的表现'
        ))

        fig.add_trace(go.Scatterpolar(
            r=reference_values,
            theta=categories,
            fill='toself',
            name='专业参考'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(player_values), max(reference_values)) * 1.2]
                )),
            showlegend=True,
            title="能力雷达图"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**移动轨迹**")

            if result.trajectory:
                # Create trajectory plot
                plotter = TrajectoryPlotter()

                # Generate static trajectory visualization
                traj_viz = plotter.draw_static_trajectory(
                    result.trajectory,
                    size=(400, 400),
                )

                st.image(traj_viz, use_container_width=True)

        with col2:
            if show_heatmap and result.heatmap is not None:
                st.markdown("**步法热力图**")

                # Create heatmap visualization
                fig = px.imshow(
                    result.heatmap,
                    color_continuous_scale='hot',
                    title="场上活动分布"
                )
                fig.update_layout(
                    xaxis_title="X",
                    yaxis_title="Y",
                    coloraxis_colorbar=dict(title="强度")
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        display_custom_metrics(result.metrics)

    with tab5:
        display_advanced_metrics(result)

    # Recommendations
    if result.recommendations:
        st.subheader("💡 个性化改进建议")

        # Priority colors and labels
        priority_colors = {
            "critical": ("#dc2626", "🔴 紧急"),  # Red
            "high": ("#ea580c", "🟠 重要"),      # Orange
            "medium": ("#ca8a04", "🟡 建议"),    # Yellow
            "low": ("#16a34a", "🟢 提示"),       # Green
        }

        # Show recommendation count
        st.caption(f"共 {len(result.recommendations)} 项建议，已按优先级排序")

        for i, rec in enumerate(result.recommendations, 1):
            priority = rec.get('priority', 'low')
            color, label = priority_colors.get(priority, ("#6b7280", "⚪"))

            # Build metric comparison display if available
            metric_info = ""
            if 'player_value' in rec and 'reference_value' in rec:
                metric = rec.get('metric', '')
                player_val = rec['player_value']
                ref_val = rec['reference_value']

                # Format based on metric type
                if 'time' in metric or 'response' in metric:
                    metric_info = f"<br><small>📊 您的数据: {player_val:.3f}s | 专业参考: {ref_val:.3f}s</small>"
                elif 'speed' in metric or 'frequency' in metric:
                    metric_info = f"<br><small>📊 您的数据: {player_val:.2f} | 专业参考: {ref_val:.2f}</small>"
                elif 'ratio' in metric or 'efficiency' in metric:
                    metric_info = f"<br><small>📊 您的数据: {player_val:.1%} | 专业参考: {ref_val:.1%}</small>"
                elif isinstance(player_val, (int, float)) and isinstance(ref_val, (int, float)):
                    if player_val < 10:  # Likely a ratio or small number
                        metric_info = f"<br><small>📊 您的数据: {player_val:.2f} | 专业参考: {ref_val:.2f}</small>"
                    else:
                        metric_info = f"<br><small>📊 您的数据: {player_val:.1f} | 专业参考: {ref_val:.1f}</small>"

            with st.container():
                st.markdown(f"""
                <div class="recommendation-box" style="border-left: 4px solid {color}; padding-left: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong style="font-size: 1.1em;">{i}. {rec['area']}</strong>
                        <span style="color: {color}; font-weight: bold; font-size: 0.9em;">{label}</span>
                    </div>
                    <div style="margin-bottom: 8px; color: #374151;">
                        <strong>问题：</strong>{rec['issue']}
                    </div>
                    <div style="background-color: #f3f4f6; padding: 10px; border-radius: 6px; color: #1f2937; line-height: 1.6;">
                        <strong>💡 建议：</strong>{rec['recommendation']}
                    </div>
                    {metric_info}
                </div>
                <div style="margin-bottom: 12px;"></div>
                """, unsafe_allow_html=True)


def opponent_analysis_page(reference_level: str, model_complexity: int):
    """Opponent analysis and strategy recommendation page"""

    st.markdown("### 🎯 对手分析")
    st.markdown("""
    通过分析对手的比赛视频，了解其战术特点和技术弱点，制定针对性对战策略：
    - 📊 分析对手的击球习惯和落点分布
    - 🔍 识别对手的技术弱点
    - 🏆 生成对战策略建议

    > 💡 对手分析会自动使用标准模式进行分析，无需选择参考水平。
    """)
    st.divider()

    # Initialize session state
    if 'opponent_analysis_done' not in st.session_state:
        st.session_state.opponent_analysis_done = False
    if 'opponent_result' not in st.session_state:
        st.session_state.opponent_result = None

    # File upload
    uploaded_file = st.file_uploader(
        "上传对手比赛视频",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="上传对手的比赛视频，系统将分析其战术特点"
    )

    # Reset when new file
    if uploaded_file is not None:
        current_file_name = uploaded_file.name
        if st.session_state.get('opponent_last_file') != current_file_name:
            st.session_state.opponent_analysis_done = False
            st.session_state.opponent_result = None
            st.session_state.opponent_last_file = current_file_name
            st.rerun()

    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📹 对手视频预览")
            st.video(video_path)

        with col2:
            st.subheader("⚙️ 分析设置")
            analyze_shots = st.checkbox("分析击球习惯", value=True, help="分析对手喜欢击球的位置和类型")
            analyze_movement = st.checkbox("分析移动模式", value=True, help="分析对手的移动特点和覆盖范围")
            generate_strategy = st.checkbox("生成策略建议", value=True, help="根据分析结果生成对战策略")

            st.divider()

            if not st.session_state.opponent_analysis_done:
                if st.button("🔍 开始分析对手", type="primary", use_container_width=True):
                    try:
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def progress_callback(frame_idx: int, total_frames: int):
                            progress = min(1.0, frame_idx / max(1, total_frames - 1))
                            progress_bar.progress(progress)
                            status_text.text(f"正在分析... {frame_idx}/{total_frames} 帧 ({progress*100:.1f}%)")

                        # Create analyzer - 使用中级水平（对手分析只需分析战术模式）
                        analyzer = BadmintonAnalyzer(
                            model_complexity=model_complexity,
                            enable_smoothing=True,
                            reference_level="intermediate",
                        )

                        # Process video
                        status_text.text("正在分析对手视频，请稍候...")
                        result = analyzer.process_video(
                            video_path=video_path,
                            show_progress=True,
                            progress_callback=progress_callback,
                        )

                        progress_bar.empty()
                        status_text.empty()

                        st.session_state.opponent_result = result
                        st.session_state.opponent_analysis_done = True
                        st.success("✅ 分析完成！")

                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"分析失败: {str(e)}")
                        import traceback
                        with st.expander("查看详细错误"):
                            st.code(traceback.format_exc())

        # Show analysis results
        if st.session_state.opponent_analysis_done and st.session_state.opponent_result:
            result = st.session_state.opponent_result

            st.divider()
            st.markdown("## 📊 对手分析结果")

            # Basic metrics
            metrics = result.metrics
            score = result.efficiency_score

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("综合评分", f"{score.overall:.1f}")
            with col2:
                st.metric("击球次数", len(result.shots))
            with col3:
                st.metric("总步数", metrics.total_steps)
            with col4:
                st.metric("最大速度", f"{metrics.max_speed:.2f} m/s")

            # Tactical analysis
            if analyze_shots and result.shots:
                st.markdown("### 🎯 击球习惯分析")

                # Shot type distribution
                shot_types = {}
                for shot in result.shots:
                    shot_type = shot.shot_type.value
                    shot_types[shot_type] = shot_types.get(shot_type, 0) + 1

                if shot_types:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**击球类型分布**")
                        shot_df = pd.DataFrame(
                            list(shot_types.items()),
                            columns=['击球类型', '次数']
                        )
                        shot_df = shot_df.sort_values('次数', ascending=False)
                        fig = px.bar(
                            shot_df,
                            x='击球类型',
                            y='次数',
                            color='次数',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.markdown("**主要击球类型**")
                        for shot_type, count in sorted(shot_types.items(), key=lambda x: x[1], reverse=True):
                            pct = count / len(result.shots) * 100
                            st.write(f"  {shot_type}: {count}次 ({pct:.1f}%)")

                # Preferred court zones
                if hasattr(result, 'tactical_geometry') and result.tactical_geometry:
                    tg = result.tactical_geometry
                    if tg.preferred_zones:
                        st.markdown("**常击球区域**")
                        zones_str = ", ".join(tg.preferred_zones[:5])
                        st.info(f"对手偏好区域: {zones_str}")

            # Movement analysis
            if analyze_movement:
                st.markdown("### 🏃 移动模式分析")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("上网速度", f"{metrics.net_approach_speed:.2f} m/s")
                with col2:
                    st.metric("后退速度", f"{metrics.backward_speed:.2f} m/s")
                with col3:
                    st.metric("横向速度", f"{metrics.lateral_speed:.2f} m/s")

                # Movement assessment
                speed_issues = []
                if metrics.net_approach_speed < 0.4:
                    speed_issues.append("上网速度较慢")
                if metrics.backward_speed < 0.3:
                    speed_issues.append("后退速度较慢")
                if metrics.lateral_speed < 0.35:
                    speed_issues.append("横向移动较慢")

                if speed_issues:
                    st.markdown("**移动弱点识别**")
                    for issue in speed_issues:
                        st.warning(f"⚠️ {issue}")

            # Strategy recommendations
            if generate_strategy:
                st.markdown("### 🏆 对战策略建议")

                recommendations = []

                # Based on shot analysis
                if result.shots:
                    shot_types = {}
                    for shot in result.shots:
                        shot_type = shot.shot_type.value
                        shot_types[shot_type] = shot_types.get(shot_type, 0) + 1

                    # Analyze weaknesses based on shot types
                    most_common = max(shot_types.items(), key=lambda x: x[1])[0] if shot_types else None

                    if most_common == "CLEAR":
                        recommendations.append({
                            "title": "针对高远球",
                            "description": "对手偏好打高远球，建议多打吊球和网前球，限制其进攻机会。",
                            "priority": "high"
                        })
                    elif most_common == "DROP":
                        recommendations.append({
                            "title": "针对吊球",
                            "description": "对手擅长吊球，建议提高网前防守准备速度，多打平抽球对抗。",
                            "priority": "high"
                        })
                    elif most_common == "SMASH":
                        recommendations.append({
                            "title": "针对杀球",
                            "description": "对手杀球威胁大，建议加强防守准备，减少半场球给对手进攻机会。",
                            "priority": "high"
                        })

                # Based on movement analysis
                if metrics.net_approach_speed < 0.4:
                    recommendations.append({
                        "title": "攻击上网弱点",
                        "description": "对手上网速度慢，建议多吊网前球，迫使其上网，然后打对角线或平抽球。",
                        "priority": "medium"
                    })

                if metrics.backward_speed < 0.3:
                    recommendations.append({
                        "title": "攻击后退弱点",
                        "description": "对手后退速度慢，建议多打高远球到后场，迫使其后退，然后吊网前或杀球。",
                        "priority": "medium"
                    })

                if metrics.lateral_speed < 0.35:
                    recommendations.append({
                        "title": "攻击横向弱点",
                        "description": "对手横向移动慢，建议多打对角线球，调动对手在场地两侧移动。",
                        "priority": "medium"
                    })

                # Based on court coverage
                if metrics.coverage_ratio < 0.25:
                    recommendations.append({
                        "title": "扩大场地使用",
                        "description": "对手场地覆盖范围有限，建议多打大角度球，消耗其体力。",
                        "priority": "low"
                    })

                # Add general strategy
                recommendations.append({
                    "title": "控制节奏",
                    "description": "根据场上形势灵活调整节奏，不要被对手节奏带走。",
                    "priority": "low"
                })

                # Display recommendations
                priority_colors = {
                    "high": "#ef4444",
                    "medium": "#f59e0b",
                    "low": "#22c55e"
                }

                for i, rec in enumerate(recommendations, 1):
                    color = priority_colors.get(rec['priority'], "#6b7280")
                    st.markdown(f"""
                    <div style="background-color: #f9fafb; border-radius: 8px; padding: 16px; margin-bottom: 12px; border-left: 4px solid {color};">
                        <div style="font-size: 1.1rem; font-weight: bold; color: #1f2937; margin-bottom: 8px;">
                            {i}. {rec['title']} <span style="font-size: 0.8rem; color: {color};">[{rec['priority'].upper()}]</span>
                        </div>
                        <div style="color: #4b5563; line-height: 1.6;">
                            {rec['description']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Heatmap
            if result.heatmap is not None:
                st.markdown("### 🗺️ 对手场地覆盖热力图")
                fig = px.imshow(
                    result.heatmap,
                    color_continuous_scale='hot',
                    title="对手场上活动分布"
                )
                fig.update_layout(
                    xaxis_title="X",
                    yaxis_title="Y",
                    coloraxis_colorbar=dict(title="强度")
                )
                st.plotly_chart(fig, use_container_width=True)


def player_selection_page(reference_level: str, model_complexity: int):
    """Player selection and tracking page"""

    st.markdown("### 👥 球员选择分析")
    st.markdown("""
    当视频中有多个球员时，可以选择要分析的特定球员：
    - 🔍 自动检测视频中的所有球员
    - 👆 点击选择要分析的球员
    - 🎯 针对特定球员进行步法分析
    """)
    st.divider()

    # Initialize session state for player selection
    if 'detected_players' not in st.session_state:
        st.session_state.detected_players = None
    if 'selected_player_id' not in st.session_state:
        st.session_state.selected_player_id = None
    if 'player_video_path' not in st.session_state:
        st.session_state.player_video_path = None

    # File upload
    uploaded_file = st.file_uploader(
        "上传包含多个球员的比赛视频",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="支持 MP4, AVI, MOV, MKV 格式"
    )

    # Reset state when new file is uploaded
    if uploaded_file is not None:
        # Check if this is a new file (different from previous)
        current_file_name = uploaded_file.name
        if st.session_state.get('last_uploaded_file') != current_file_name:
            # Reset all detection-related state for new file
            st.session_state.detected_players = None
            st.session_state.selected_player_id = None
            st.session_state.detection_preview = None
            st.session_state.start_player_analysis = False
            st.session_state.last_uploaded_file = current_file_name
            st.rerun()  # Force rerun to refresh UI

    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
            st.session_state.player_video_path = video_path

        # Player detection section
        if st.session_state.detected_players is None:
            st.info("👆 点击下方按钮检测视频中的球员")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📹 视频预览")
                st.video(video_path)

            with col2:
                st.subheader("🎛️ 操作")

                if st.button("🔍 检测球员", type="primary", use_container_width=True):
                    with st.spinner("正在检测球员..."):
                        try:
                            import mediapipe as mp

                            # Read first frame
                            cap = cv2.VideoCapture(video_path)
                            ret, frame = cap.read()
                            cap.release()

                            if not ret:
                                st.error("无法读取视频")
                                return

                            # Detect players with fallback
                            mp_pose = mp.solutions.pose
                            mp_drawing = mp.solutions.drawing_utils
                            mp_drawing_styles = mp.solutions.drawing_styles

                            # Try requested complexity, fallback to 1 if needed
                            complexities_to_try = [model_complexity]
                            if model_complexity == 2:
                                complexities_to_try = [2, 1]

                            detection_success = False
                            last_error = None

                            for complexity in complexities_to_try:
                                pose = None
                                try:
                                    # Create pose detector
                                    pose = mp_pose.Pose(
                                        static_image_mode=True,
                                        model_complexity=complexity,
                                        min_detection_confidence=0.5
                                    )

                                    # Test with actual frame
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    results = pose.process(frame_rgb)

                                    if results.pose_landmarks:
                                        # Detected at least one player
                                        st.session_state.detected_players = [{
                                            'id': 0,
                                            'name': '球员 1',
                                            'confidence': 0.95
                                        }]

                                        # Create annotated image
                                        annotated_frame = frame.copy()
                                        h, w = frame.shape[:2]

                                        # Draw landmarks
                                        mp_drawing.draw_landmarks(
                                            annotated_frame,
                                            results.pose_landmarks,
                                            mp_pose.POSE_CONNECTIONS,
                                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                                        )

                                        # Convert to RGB for display
                                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                                        st.session_state.detection_preview = annotated_frame_rgb

                                        if complexity < model_complexity:
                                            st.warning(f"⚠️ 模型精度 {model_complexity} 不支持，已自动使用精度 {complexity}")

                                        st.success("✅ 检测到 1 个球员")
                                        detection_success = True
                                        break

                                except RuntimeError as e:
                                    last_error = e
                                    if "InferenceCalculator" in str(e) and complexity < complexities_to_try[-1]:
                                        # Try lower complexity - don't raise, just continue to next
                                        pass
                                    else:
                                        # Can't use this complexity at all
                                        pass
                                finally:
                                    # Always try to close pose detector, but don't fail if it errors
                                    if pose is not None:
                                        try:
                                            pose.close()
                                        except:
                                            pass  # Ignore close errors

                            # If no player detected after all attempts
                            if not detection_success:
                                if last_error and "InferenceCalculator" in str(last_error):
                                    st.error(f"❌ 模型初始化失败。建议：使用模型精度 0 或 1")
                                else:
                                    st.warning("⚠️ 未检测到球员，请确保视频中人物清晰可见")

                        except Exception as e:
                            st.error(f"检测失败: {str(e)}")
                            import traceback
                            with st.expander("查看详细错误"):
                                st.code(traceback.format_exc())

        # Show detection results and player selection
        if st.session_state.detected_players is not None:
            st.divider()

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("🔍 检测结果")

                if hasattr(st.session_state, 'detection_preview'):
                    st.image(st.session_state.detection_preview, use_container_width=True, caption="球员检测预览")

            with col2:
                st.subheader("👤 选择球员")

                for player in st.session_state.detected_players:
                    player_id = player['id']
                    player_name = player['name']

                    if st.button(f"{player_name}", key=f"select_player_{player_id}", use_container_width=True):
                        st.session_state.selected_player_id = player_id
                        st.success(f"✅ 已选择 {player_name}")

                st.divider()

                # Analysis options for selected player
                if st.session_state.selected_player_id is not None:
                    st.markdown(f"**当前选择**: 球员 {st.session_state.selected_player_id + 1}")

                    generate_annotated = st.checkbox("生成标注视频", value=True)
                    show_heatmap = st.checkbox("生成热力图", value=True)

                    st.divider()

                    if st.button("🚀 开始分析", type="primary", use_container_width=True):
                        st.session_state.start_player_analysis = True

            # Process video if analysis started
            if st.session_state.get('start_player_analysis', False):
                with st.spinner("正在分析..."):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def progress_callback(frame_idx, total_frames):
                            progress = min(1.0, frame_idx / max(1, total_frames - 1))
                            progress_bar.progress(progress)
                            status_text.text(f"处理中... {frame_idx}/{total_frames} 帧 ({progress*100:.1f}%)")

                        output_path = None
                        if generate_annotated:
                            output_dir = Path(__file__).parent.parent / "data" / "output"
                            output_dir.mkdir(parents=True, exist_ok=True)
                            output_path = str(output_dir / f"player_{st.session_state.selected_player_id}_{int(time.time())}.mp4")

                        with BadmintonAnalyzer(
                            model_complexity=model_complexity,
                            reference_level=reference_level,
                        ) as analyzer:
                            result = analyzer.process_video(
                                video_path=st.session_state.player_video_path,
                                output_path=output_path,
                                progress_callback=progress_callback,
                            )

                        progress_bar.empty()
                        status_text.empty()
                        st.session_state.start_player_analysis = False

                        st.success("✅ 分析完成！")
                        display_results(result, show_heatmap)

                    except Exception as e:
                        st.error(f"分析失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())


def live_analysis_page(reference_level: str, model_complexity: int):
    """Live camera analysis page"""

    st.markdown("### 📷 实时分析")
    st.markdown("""
    使用摄像头进行实时步法分析：
    - 🔴 实时姿态估计和重心追踪
    - 📊 实时指标显示
    - ⚡ 即时反馈和建议
    """)
    st.divider()

    st.warning("⚠️ 实时分析功能需要摄像头访问权限")

    # Camera selection
    camera_id = st.selectbox(
        "选择摄像头",
        options=[0, 1, 2],
        format_func=lambda x: f"摄像头 {x}"
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📷 实时视频")
        frame_placeholder = st.empty()

    with col2:
        st.subheader("📊 实时数据")
        metrics_placeholder = st.empty()

    if st.button("▶️ 开始实时分析", type="primary"):
        run_live_analysis(camera_id, frame_placeholder, metrics_placeholder, reference_level, model_complexity)


def run_live_analysis(camera_id: int, frame_placeholder, metrics_placeholder, reference_level: str, model_complexity: int):
    """Run live camera analysis"""

    try:
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            st.error(f"无法打开摄像头 {camera_id}")
            return

        # Create analyzer
        analyzer = BadmintonAnalyzer(
            model_complexity=model_complexity,
            reference_level=reference_level,
        )

        stop_button = st.button("⏹️ 停止分析")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            annotated_frame, result = analyzer.process_frame_realtime(frame)

            # Convert to RGB for Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display frame
            frame_placeholder.image(
                annotated_frame_rgb,
                channels="RGB",
                use_container_width=True
            )

            # Update metrics if available
            if result:
                with metrics_placeholder.container():
                    score = result.efficiency_score
                    st.metric("综合评分", f"{score.overall:.0f}")
                    st.metric("步数", result.metrics.total_steps)
                    st.metric("距离", f"{result.metrics.total_distance:.1f}m")

            # Check for stop
            if stop_button:
                break

            time.sleep(0.03)  # ~30 fps

        cap.release()

    except Exception as e:
        st.error(f"实时分析出错: {e}")


if __name__ == "__main__":
    main()
