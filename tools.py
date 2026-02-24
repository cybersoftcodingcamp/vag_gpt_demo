# tools.py (Module for Tools and Utilities)
import base64
import os
import magic
import requests
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.tools import tool
from ultralytics import YOLO
from typing import Type

# Load YOLO model
yolo_model = YOLO("yolo11x.pt")

# Encode image function
def encode_image(image_path_or_url: str, get_mime_type: bool = False):
    if image_path_or_url.startswith("http"):
        try:
            response = requests.get(image_path_or_url, stream=True)
            response.raise_for_status()
            image = response.content
            mime_type = response.headers.get("content-type", None)
            base64_encoded = base64.b64encode(image).decode('utf-8')
            if get_mime_type:
                return base64_encoded, mime_type
            else:
                return base64_encoded
        except requests.exceptions.RequestException as e:
            return None, None if get_mime_type else None
    else:
        if not os.path.exists(image_path_or_url):
            return None, None if get_mime_type else None
        mime_type = magic.Magic(mime=True).from_file(image_path_or_url)
        if mime_type.startswith("image/"):
            with open(image_path_or_url, "rb") as image_file:
                if get_mime_type:
                    return base64.b64encode(image_file.read()).decode("utf-8"), mime_type
                else:
                    return base64.b64encode(image_file.read()).decode("utf-8")
        else:
            return None, None if get_mime_type else None

# ImageInput and Extractor Chain
class ImageInput(BaseModel):
    image_path_or_url: str = Field(description="Image path or URL")

parser = PydanticOutputParser(pydantic_object=ImageInput)
prompt = ChatPromptTemplate.from_template(
    "Extract the image path or URL from the following input:\n\n{input}\n\n{format_instructions}"
).partial(format_instructions=parser.get_format_instructions())
extractor_chain = prompt | ChatOpenAI(model="gpt-4o-mini") | parser

# Image Description
class ImageDescription(BaseModel):
    image_description: str = Field(description="Detailed description of the image")

def image_describer_prompt_func(inputs: dict):
    image_path_or_url = inputs["image_path_or_url"]
    image_b64, image_mime_type = encode_image(image_path_or_url, get_mime_type=True)
    image_describer_chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""Bạn là một chuyên gia mô tả hình ảnh với khả năng phân tích nâng cao, tập trung vào việc trả lời query của người dùng một cách phù hợp nhất dựa trên phân tích hình ảnh.\n\n
Khi được trình bày một hình ảnh và query (nếu có), hãy cung cấp mô tả chi tiết, chính xác và khách quan về nội dung nhìn thấy, nhưng ưu tiên tập trung vào các khía cạnh liên quan trực tiếp đến query để đưa ra câu trả lời phù hợp nhất. Cấu trúc phản hồi của bạn như một báo cáo toàn diện, bao quát các khía cạnh sau một cách sâu sắc, nhưng chỉ nếu chúng liên quan đến query:

Đảm bảo mô tả toàn diện nhưng ngắn gọn, ưu tiên theo mức độ liên quan đến query (yếu tố phù hợp với query trước tiên). Tránh bất kỳ thông tin nào không nhìn thấy hoặc có thể suy luận hợp lý; không suy đoán, thêm ý kiến cá nhân, hoặc tưởng tượng chi tiết. Nếu hình ảnh không rõ ràng ở một phần, hãy ghi chú rõ ràng (ví dụ: 'khu vực mờ ở góc dưới bên phải ngăn cản việc nhận dạng'). Nếu query không chỉ định, cung cấp mô tả tổng quát nhưng vẫn tập trung vào các yếu tố chính."""),
        HumanMessage(content=[
            {"type": "text", "text": "Mô tả hình ảnh sau cho tôi:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime_type};base64,{image_b64}", "detail": "high"}  # Thay 'low' bằng 'high' để phân tích chi tiết hơn
            }
        ])
    ])
    return image_describer_chat_template.invoke({})

image_describer_agent = image_describer_prompt_func | ChatOpenAI(model="gpt-4o-mini").with_structured_output(ImageDescription)

# Image Describer Tool
class ImageDescriberInput(BaseModel):
    text: str = Field(description="Path or URL to the image in the format PNG or JPG/JPEG")

class ImageDescriberTool(BaseTool):
    name: str = "image_describer"
    description: str = "This tool can describe the image in a detailed way"
    args_schema: Type[BaseModel] = ImageDescriberInput
    return_direct: bool = True

    def _run(self, text: str, run_manager: CallbackManagerForToolRun = None) -> str:
        try:
            parsed = extractor_chain.invoke({"input": text})
        except Exception as e:
            return f"Failed to extract image URL: {str(e)}"
        image_path_or_url = parsed.image_path_or_url
        if not image_path_or_url:
            return "No image URL found in the input."
        output = image_describer_agent.invoke({"image_path_or_url": image_path_or_url})
        return output.image_description

image_describer_tool = ImageDescriberTool()

# Object Detection Tool
class ObjectDetectingAndCountingInput(BaseModel):
    text: str = Field(description="Path or URL to the image in the format PNG or JPG/JPEG")

@tool(
    "detect_and_count_objects",
    description="Detect and count objects within the image. The return will be a dictionary, containing the counting dictionary (counting how many instance of each object class) and a list of dictionaries, containing the object names, confidence scores, and location in the image (in (x1, x2, y1, y2) format).",
    args_schema=ObjectDetectingAndCountingInput
)
def detect_and_count_object_tool(text: str):
    try:
        parsed = extractor_chain.invoke({"input": text})
    except Exception as e:
        return f"Failed to extract image URL: {str(e)}"
    image_path_or_url = parsed.image_path_or_url
    if not image_path_or_url:
        return "No image URL found in the input."

    results = yolo_model(image_path_or_url, verbose=False)

    detections = []
    counting = {}

    for result in results:
        boxes = result.boxes
        class_names = result.names

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2)
            })

            counting[class_name] = counting.get(class_name, 0) + 1

    return str({'counting': counting, 'detections': detections})