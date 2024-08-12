import os
import base64
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain.schema import HumanMessage  # Assuming you are using langchain.schema for HumanMessage

# Set environment variables
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Auto-ConfigFee"
os.environ["TAVILY_API_KEY"] = "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"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/leduy/PycharmProjects/Funny_days_wRAG/credentials/gen-lang-client-0882885035-7a0a41538ab7.json"

# Initialize the Model
project = "gen-lang-client-0882885035"
location = "europe-west1"
claude35_sonnet = ChatAnthropicVertex(
    model_name="claude-3-5-sonnet@20240620",
    project=project,
    location=location,
    max_output_tokens=4096,
)

# Function to convert image file to base64 encoded data
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# IMG PATH LOAD
# image_path = "data/detected/train/333/6-lon-bia-333-330ml-202307281705381867_300x300.jpg"
# image_path2 = "data/detected/train/lac_viet/0543bd173277619.65cb1df4307c2.png"
# image_path3 = "data/original/train/saigon_export_premium/6-lon-bia-sai-gon-export-premium-330ml-202307271017554170.jpg"
# image_path4 = "latest_picture/img.png"
# # Convert each image
# image_data = image_to_base64(image_path)
# image_data2 = image_to_base64(image_path2)
# image_data3 = image_to_base64(image_path3)
# image_data4 = image_to_base64(image_path4)
# # Print or use the base64 encoded data
# print(image_data)
# print(image_data2)
# print(image_data3)


# URL LOAD
import base64
import httpx

image_url = "https://cdn.tuoitre.vn/thumb_w/480/2021/9/30/huda-1-16329958956931056907531.jpg"
# image_url2 = "https://static.paysmart.com.vn/store/202405/0931515069_30663166691125877.png"
# image_url3 = "https://static.paysmart.com.vn/store/202405/0931515069_30663166691258981.png"
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
# image_data2 = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
# image_data3 = base64.b64encode(httpx.get(image_url).content).decode("utf-8")


# Create the message
message = HumanMessage(
    content=[
        {"type": "text", "text": """bia gì đây, trả lời tên của loại bia chuẩn
"""},
        # {
        #     "type": "image_url",
        #     "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        # },
        # {
        #     "type": "image_url",
        #     "image_url": {"url": f"data:image/png;base64,{image_data2}"},
        # },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        }
    ],
)

# Invoke the model and print the response
response = claude35_sonnet.invoke([message])
print(response.content)
