"""
Simple test Streamlit app
"""
import streamlit as st

st.set_page_config(
    page_title="测试",
    layout="wide",
)

st.title("🏸 羽毛球步法效率分析系统 - 测试")
st.write("如果你能看到这个页面，说明 Streamlit 运行正常！")

st.info("👆 这是一个测试页面")

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.header("列 1")
    st.write("这是第一列的内容")

with col2:
    st.header("列 2")
    st.write("这是第二列的内容")

st.divider()

uploaded_file = st.file_uploader("上传测��文件", type=['txt', 'md'])

if uploaded_file:
    st.success(f"文件已上传: {uploaded_file.name}")
    st.write("文件内容:")
    st.text(str(uploaded_file.read()))
else:
    st.info("请上传一个文件")

st.divider()
st.markdown("---")
st.write("✅ 测试完成！如果你能看到这个，界面是正常的。")
