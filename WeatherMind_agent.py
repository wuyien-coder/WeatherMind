from dataclasses import dataclass
import requests
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

# 加载环境变量
load_dotenv()

# 定义系统提示 - 这个嘛从用户角度思考
SYSTEM_PROMPT = """你是一位专业的 AI 天气助手，具备以下核心能力：

【工具使用】
1. get_user_location：基于 IP 的多来源定位工具（自动故障转移）
2. get_weather_for_location：全球天气数据查询（实时 +用户需求天预报）

【产品功能设计】
- 智能意图识别：自动判断用户需要实时天气、短期预报还是长期趋势
- 场景化服务：根据天气提供穿衣、出行、运动等个性化建议
- 极端天气预警：主动提醒高温、暴雨、大雪等危险天气
- 多轮对话记忆：支持连续追问和上下文理解

【数据处理规范】
- weathercode 映射：0=晴，1=主要晴，2=多云，3=阴，61/63/65=雨，71/73/75=雪，95=雷雨
- 温度单位：摄氏度 (°C)，风速单位：km/h
- 降水概率：percentage_max 字段表示最大降水概率

【用户体验原则】
- 结构化呈现：先完整数据，再精炼总结
- 个性化交互：根据用户问题类型调整回复粒度
- 容错处理：定位失败时友好引导用户提供城市名"""


# 定义上下文模式 - 支持用户画像和会话管理
@dataclass
class Context:
    """运行时上下文：支持多用户会话管理和个性化服务
    暂没真正使用，用于功能扩展"""
    user_id: str  # 用户唯一标识
    preferences: Optional[Dict[str, Any]] = None  # 用户偏好配置


# 定义工具
@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气数据（实时 +7 天预报）
    
    技术实现：
    - API 提供商：Open-Meteo（免费、无需认证）
    - 数据维度：温度、降水概率、天气状况、风速等
    - 更新频率：每小时更新
    
    Args:
        city: 城市名称（支持中英文）
    
    Returns:
        JSON 格式的结构化数据，包含：
        - city: 城市名称
        - coordinates: 经纬度坐标
        - current: 实时天气数据
        - daily_forecast: 7 天逐日预报
    """
    try:
        # Step 1: 地理编码（将城市名转换为经纬度）
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=zh&format=json"
        geo_response = requests.get(geocode_url, timeout=5)
        geo_data = geo_response.json()

        if not geo_data.get('results'):
            return json.dumps({"error": f"未找到城市 '{city}' 的信息"}, ensure_ascii=False)

        location = geo_data['results'][0]
        lat = location['latitude']
        lon = location['longitude']
        city_name = location['name']

        # Step 2: 获取天气预报数据（包含实时 +最大天数 天预报）
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
            f"&timezone=auto&forecast_days=16"
        )
        weather_response = requests.get(weather_url, timeout=5)
        weather_data = weather_response.json()

        # Step 3: 构建结构化响应数据
        result = {
            "city": city_name,
            "coordinates": {"lat": lat, "lon": lon},
            "current": weather_data.get('current_weather', {}),
            "daily_forecast": weather_data.get('daily', {}),
            "timestamp": datetime.now().isoformat()
        }

        return json.dumps(result, ensure_ascii=False)

    except requests.exceptions.Timeout:
        return json.dumps({"error": "天气服务请求超时"}, ensure_ascii=False)
    except requests.exceptions.ConnectionError:
        return json.dumps({"error": "无法连接天气服务"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"获取天气失败：{str(e)}"}, ensure_ascii=False)


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """基于 IP 地址的用户定位工具（多源故障转移机制）
    
    技术亮点：
    - 多 API 源备份：ipapi.co / ip-api.com / ipinfo.io
    - 自动故障转移：一个失败自动尝试下一个
    - 错误追踪：记录每个服务的失败原因
    - 超时控制：5 秒超时避免长时间等待
    
    Returns:
        成功：返回城市名称
        失败：返回"定位失败"并打印详细错误日志
    """
    # 多源定位服务配置（按优先级排序）
    location_services = [
        {"name": "ip-api.com", "url": "http://ip-api.com/json/?lang=zh-CN"},
        {"name": "ipapi.co", "url": "https://ipapi.co/json/"},
        {"name": "ipinfo.io", "url": "https://ipinfo.io/json"},
    ]

    errors = []

    for service in location_services:
        try:
            response = requests.get(service["url"], timeout=5)
            location_data = response.json()

            # 根据不同 API 的响应格式进行适配
            if service['name'] == 'ip-api.com':
                if location_data.get('status') == 'fail':
                    errors.append(f"{service['name']}: {location_data.get('message', '未知错误')}")
                    continue
                city = location_data.get('city', '')
            elif service['name'] == 'ipinfo.io':
                if 'error' in location_data:
                    errors.append(f"{service['name']}: {location_data.get('error', '未知错误')}")
                    continue
                city = location_data.get('city', '')
            else:  # ipapi.co
                if 'error' in location_data:
                    errors.append(f"{service['name']}: {location_data.get('error', '未知错误')}")
                    continue
                city = location_data.get('city', '')

            # 验证城市信息有效性
            if not city or city == '定位出错':
                errors.append(f"{service['name']}: 未获取到有效城市信息")
                continue

            print(f"✅ 定位成功 ({service['name']}): {city}")
            return city

        except requests.exceptions.Timeout:
            errors.append(f"{service['name']}: 请求超时 (5s)")
            continue
        except requests.exceptions.ConnectionError:
            errors.append(f"{service['name']}: 网络连接失败")
            continue
        except Exception as e:
            errors.append(f"{service['name']}: {str(e)}")
            continue

    # 所有服务均失败时的错误报告
    print(f"\n⚠️ 定位服务不可用（共 {len(errors)} 个错误）:")
    for i, err in enumerate(errors, 1):
        print(f"   {i}. {err}")

    return "定位失败"


# 配置大语言模型 - 使用阿里云通义千问 3.5 Plus
# 技术选型理由：
# - 支持 Function Calling（工具调用）
# - 中文理解能力强
# - 性价比高（便宜好用，适合测试项目）
model = init_chat_model(
    "qwen3.5-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_provider="openai",
    temperature=0.3,  # 平衡创造性与准确性
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量读取 API 密钥
    extra_body={"enable_thinking": False}  # 禁用思考模式以兼容 tool_choice
)


# 设置会话记忆模块
checkpointer = InMemorySaver()  # 内存级检查点，支持多轮对话

# 创建 Agent - LangGraph 架构
# 核心组件：
# - LLM: qwen3.5-plus（核心：推理引擎）
# - Tools: 定位 + 天气（执行能力）
# - Memory: 多轮对话记忆
# - Context: 用户上下文管理

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    checkpointer=checkpointer,

)

# 会话配置 - 支持多用户并行对话
config = {"configurable": {"thread_id": "1"}}  # thread_id 隔离不同会话

def main():
    """主函数：交互式天气助手"""
    print("="*60)
    print("🤖 WeatherMind AI 天气助手 v1.0")
    print("   基于 LangChain + LangGraph 构建的智能体")
    print("   功能：智能定位 | 实时天气 | 7 天预报 | 场景化建议")
    print("="*60)
    print("💡 输入 'help' 查看使用示例，输入 'quit' 退出\n")

    while True:
        try:
            user_input = input("👤 你：").strip()

            # 空输入处理
            if not user_input:
                continue

            # 退出命令
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n👋 感谢使用 WeatherMind！祝你有美好的一天！")
                break

            # 帮助信息
            if user_input.lower() in ['help', '帮助', 'h']:
                print("\n📖 使用示例：")
                print("   • '今天天气怎么样？' - 查询当前位置天气")
                print("   • '北京下周会下雨吗？' - 查询指定城市预报")
                print("   • '我需要带伞吗？' - 场景化咨询")
                print("   • '对比北京和上海的天气' - 多城市对比\n")
                continue

            # 调用 Agent 处理用户请求
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                context=Context(user_id="1")
            )

            # 解析并展示响应
            ai_message = response['messages'][-1].content
            print(f"\n🌤️ AI: {ai_message}\n")

        except KeyboardInterrupt:
            print("\n\n👋 程序中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误：{type(e).__name__} - {str(e)}\n")
            # 调试信息（开发阶段）
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

