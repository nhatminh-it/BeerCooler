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

# Replace these with the paths to your local image files
image_path = "data/detected/train/333/6-lon-bia-333-330ml-202307281705381867_300x300.jpg"
image_path2 = "data/detected/train/lac_viet/0543bd173277619.65cb1df4307c2.png"
image_path3 = "data/original/train/saigon_export_premium/6-lon-bia-sai-gon-export-premium-330ml-202307271017554170.jpg"
image_path4 = "latest_picture/img.png"
# Convert each image
image_data = image_to_base64(image_path)
image_data2 = image_to_base64(image_path2)
image_data3 = image_to_base64(image_path3)
image_data4 = image_to_base64(image_path4)
# Print or use the base64 encoded data
print(image_data)
print(image_data2)
print(image_data3)

# Create the message
message = HumanMessage(
    content=[
        {"type": "text", "text": """Nhiệm vụ của bạn là phân biệt có chai và lon bia trong hình này không? và nếu có bia thì là gì bia gì?
Trong list bia [333,
lac_viet,
saigon_export_premium,
saigon_gold,
saigon_larger,
saigon_special,
]
nếu chỉ có một loại bia thì trả về [tên bia trong list]
nếu có nhiều loại bia trong hình thì trả về 1 list bia [list bia trong hình]
Nếu chai và lon bia không thuộc thương hiệu nào ở trên, đánh nhãn [others]
Lưu ý là chai và lon bia.
lưu ý ** ** Mỗi hình ảnh input hãy thực hiện return một list cho mỗi ảnh kể cả khi nó trùng thương hiệu cũng trả về mỗi hình ảnh 1 list riêng biệt** **.
** ** Mỗi hình ảnh input hãy thực hiện return một list cho mỗi ảnh kể cả khi nó trùng thương hiệu cũng trả về mỗi hình ảnh 1 list riêng biệt** **.
"""},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data2}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data4}"},
        }
    ],
)

# Invoke the model and print the response
response = claude35_sonnet.invoke([message])
print(response.content)
