import streamlit as st
import os
import base64
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# ==========================================
# 1. 页面基础设置
# ==========================================
st.set_page_config(page_title="AI 智能菜谱助手", page_icon="👨‍🍳", layout="wide")
st.title("👨‍🍳 AI 智能菜谱助手")
st.write("上传你冰箱里的食材图片，或者直接输入文字，AI 大厨为你量身定制菜谱！")


# ==========================================
# 2. 定义数据模型与工具（和之前一样）
# ==========================================
class RecipeInfo(BaseModel):
    name: str = Field(description="菜名")
    ingredients: list[str] = Field(description="所需食材列表")
    nutrition: Literal["高", "中", "低"] = Field(description="营养价值评估")
    difficulty: Literal["简单", "中等", "困难"] = Field(description="制作难度")
    steps: list[str] = Field(description="简短的制作步骤")
    reason: str = Field(description="推荐或创意的理由")


@tool
def search_recipe_db(ingredients: str) -> str:
    """根据食材搜索本地数据库中的真实菜谱。"""
    if "土豆" in ingredients and "西红柿" in ingredients:
        return "数据库命中：土豆西红柿炖牛肉（营养极高，难度中等）"
    elif "鸡蛋" in ingredients:
        return "数据库命中：清炒鸡蛋（营养中等，难度简单）"
    return "数据库中未找到相关菜谱，请发挥你的创意自行设计。"


# ==========================================
# 3. 初始化 Agent（使用 cache_resource 防止重复加载）
# ==========================================
@st.cache_resource
def get_agent():
    load_dotenv(override=True)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL")

    # 注意：这里必须关闭思考模式，否则会报 400
    chat_model = init_chat_model(
        model="qwen3.8-flash",
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        extra_body={"enable_thinking": False}
    )

    checkpointer = InMemorySaver()

    system_prompt = """
    你是“小当家”，一位热情的AI大厨。
    1. 优先调用 `search_recipe_db` 查库。
    2. 查不到再发挥创意。
    3. 推荐理由要生动有趣。
    """

    agent = create_agent(
        model=chat_model,
        tools=[search_recipe_db],
        system_prompt=system_prompt,
        response_format=RecipeInfo,
        checkpointer=checkpointer
    )
    return agent


agent = get_agent()
config = {"configurable": {"thread_id": "web_session_1"}}

# ==========================================
# 4. 页面交互区
# ==========================================
# 使用两列布局
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📸 上传食材图片")
    uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="已上传的食材", use_column_width=True)

with col2:
    st.subheader("💬 补充说明（可选）")
    user_text = st.text_input("比如：我不想吃辣的，或者：这道菜需要做多久？")

# 提交按钮
if st.button("🚀 开始生成菜谱", type="primary", use_container_width=True):
    if uploaded_file is None and not user_text:
        st.warning("请至少上传一张图片或输入一段文字哦！")
    else:
        with st.spinner("AI 大厨正在飞速思考并切菜中..."):
            # 构建消息内容
            content_list = []

            # 如果有文字，加进去
            if user_text:
                content_list.append({"type": "text", "text": user_text})
            else:
                content_list.append({"type": "text", "text": "请看看图里有什么食材，能做什么菜？"})

            # 如果有图片，转码并加进去
            if uploaded_file is not None:
                img_bytes = uploaded_file.getvalue()
                base64_str = base64.b64encode(img_bytes).decode('utf-8')
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                })

            message = HumanMessage(content=content_list)

            # 调用 Agent
            response = agent.invoke({"messages": [message]}, config)

            # 渲染结果
            if "structured_response" in response:
                recipe = response["structured_response"]

                st.success("✅ 菜谱生成成功！")

                # 使用漂亮的卡片展示
                st.markdown(f"## 🍲 {recipe.name}")

                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"**🥬 食材：**\n{', '.join(recipe.ingredients)}")
                with c2:
                    st.info(f"**💪 营养：** {recipe.nutrition} | **🔥 难度：** {recipe.difficulty}")

                st.markdown("### 📝 制作步骤")
                for i, step in enumerate(recipe.steps, 1):
                    st.write(f"**步骤 {i}：** {step}")

                st.warning(f"💡 **大厨推荐理由：** {recipe.reason}")
            else:
                st.error("❌ 生成失败，AI 原始回复：")
                st.write(response["messages"][-1].content)