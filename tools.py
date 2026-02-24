# tools.py (Module cho Công cụ và Tiện ích)
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

# Tải mô hình YOLO
yolo_model = YOLO("yolo11x.pt")

# Hàm mã hóa ảnh
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

# ImageInput và Chuỗi Trích xuất
class ImageInput(BaseModel):
    image_path_or_url: str = Field(description="Đường dẫn hoặc URL ảnh")

parser = PydanticOutputParser(pydantic_object=ImageInput)
prompt = ChatPromptTemplate.from_template(
    "Extract the image path or URL from the following input:\n\n{input}\n\n{format_instructions}"
).partial(format_instructions=parser.get_format_instructions())
extractor_chain = prompt | ChatOpenAI(model="gpt-4o-mini") | parser

# Mô tả Ảnh
class ImageDescription(BaseModel):
    image_description: str = Field(description="Mô tả chi tiết về ảnh")

def image_describer_prompt_func(inputs: dict):
    image_path_or_url = inputs["image_path_or_url"]
    image_b64, image_mime_type = encode_image(image_path_or_url, get_mime_type=True)
    image_describer_chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""You are an expert image describer with advanced analytical capabilities. When presented with an image, provide a highly detailed, accurate, and objective description of its visible content. Structure your response as a comprehensive report, covering the following aspects in depth:

- **Overall Composition and Layout**: Describe the image's structure (e.g., symmetrical/asymmetrical, rule of thirds), focal points, foreground/background separation, and spatial organization (e.g., left/right/center dominance).
- **Objects and Entities**: List all primary and secondary objects/people/animals, their classifications (e.g., breed/species if inferable), quantities, sizes (relative to frame, e.g., 'occupies 30% of the image'), precise positions (e.g., 'top-left corner, centered horizontally'), and relationships/interactions (e.g., 'object A overlaps object B by 20%').
- **Colors and Lighting**: Analyze color palette (dominant hues, contrasts, saturation), lighting sources (natural/artificial, direction/shadows), and effects (e.g., high-key/low-key, glare, highlights/lowlights).
- **Textures and Materials**: Detail surface qualities (e.g., smooth/rough, glossy/matte), materials (e.g., wood/metal/fabric), and patterns (e.g., repetitive motifs, gradients).
- **Actions and Dynamics**: Describe any motion/implied movement (e.g., 'person running towards right, blurred background suggesting speed'), poses/expressions (e.g., 'smiling face with raised eyebrows indicating surprise'), and temporal elements (e.g., 'daytime scene with long shadows suggesting afternoon').
- **Context and Inferences**: Infer logical settings (e.g., 'urban street in modern city, likely evening based on lighting'), era/style (e.g., 'vintage photo from 1950s aesthetic'), and potential narratives (e.g., 'family gathering in a park')—but only if directly supported by visible evidence.
- **Text and Symbols**: Transcribe all visible text exactly (including fonts/styles), and describe any symbols/logos/icons with their meanings if obvious.
- **Technical Details**: Note image quality (e.g., resolution artifacts, noise), perspective (e.g., wide-angle distortion), and anomalies (e.g., overexposure in areas).

Ensure the description is exhaustive yet concise, prioritized by salience (most prominent elements first). Avoid any information not visible or reasonably inferable; do not speculate, add personal opinions, or hallucinate details. If the image is unclear in parts, note it explicitly (e.g., 'blurred area in bottom-right prevents identification')."""),
        HumanMessage(content=[
            {"type": "text", "text": "Describe the following image for me:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime_type};base64,{image_b64}", "detail": "high"}  # Thay 'low' bằng 'high' để phân tích chi tiết hơn
            }
        ])
    ])
    return image_describer_chat_template.invoke({})

image_describer_agent = image_describer_prompt_func | ChatOpenAI(model="gpt-4o-mini").with_structured_output(ImageDescription)

# Công cụ Mô tả Ảnh
class ImageDescriberInput(BaseModel):
    text: str = Field(description="Đường dẫn hoặc URL ảnh ở định dạng PNG hoặc JPG/JPEG")

class ImageDescriberTool(BaseTool):
    name: str = "image_describer"
    description: str = "Công cụ này có thể mô tả ảnh một cách chi tiết"
    args_schema: Type[BaseModel] = ImageDescriberInput
    return_direct: bool = True

    def _run(self, text: str, run_manager: CallbackManagerForToolRun = None) -> str:
        try:
            parsed = extractor_chain.invoke({"input": text})
        except Exception as e:
            return f"Không thể trích xuất URL ảnh: {str(e)}"
        image_path_or_url = parsed.image_path_or_url
        if not image_path_or_url:
            return "Không tìm thấy URL ảnh trong đầu vào."
        output = image_describer_agent.invoke({"image_path_or_url": image_path_or_url})
        return output.image_description

image_describer_tool = ImageDescriberTool()

# Công cụ Phát hiện và Đếm Đối tượng
class ObjectDetectingAndCountingInput(BaseModel):
    text: str = Field(description="Đường dẫn hoặc URL ảnh ở định dạng PNG hoặc JPG/JPEG")

@tool(
    "detect_and_count_objects",
    description="Phát hiện và đếm đối tượng trong ảnh. Kết quả trả về là từ điển, chứa từ điển đếm (đếm số lượng từng lớp đối tượng) và danh sách từ điển chứa tên đối tượng, điểm tin cậy, và vị trí trong ảnh (định dạng (x1, x2, y1, y2)).",
    args_schema=ObjectDetectingAndCountingInput
)
def detect_and_count_object_tool(text: str):
    try:
        parsed = extractor_chain.invoke({"input": text})
    except Exception as e:
        return f"Không thể trích xuất URL ảnh: {str(e)}"
    image_path_or_url = parsed.image_path_or_url
    if not image_path_or_url:
        return "Không tìm thấy URL ảnh trong đầu vào."

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