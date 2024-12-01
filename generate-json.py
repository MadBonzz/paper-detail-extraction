import os
import json
from llama_cpp import Llama

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.pipe.OCRPipe import OCRPipe


def extract_text(pdf_path):
    model_list = ["layoutlmv3", "yolo_v8_mfd", "unimernet_small"]
    local_image_dir, local_md_dir = "output/images", "output"
    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
    image_dir = str(os.path.basename(local_image_dir))

    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_path)

    pipe = OCRPipe(pdf_bytes, model_list, image_writer, end_page_id=0)

    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()

    pdf_info = pipe.pdf_mid_data["pdf_info"]

    md_content = pipe.pipe_mk_markdown(
        image_dir, drop_mode=DropMode.NONE, md_make_mode=MakeMode.MM_MD
    )

    if isinstance(md_content, list):
        md_content = "\n".join(md_content)
    return md_content

directory_path = "question-papers"

files = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]

user_response = None

for file in files:
    pdf_path = os.path.join(directory_path, file)
    print(f"Processing: {pdf_path}")
    user_response = extract_text(pdf_path)

print(user_response)
if len(user_response) > 300:
    user_response = user_response[:300]

llm = Llama(
    model_path="C:/Users/shour/.cache/lm-studio/models/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf", chat_format="chatml-function-calling", n_ctx=2048
)

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a data extraction assistant. Your task is to extract specific fields from a given text. "
                "Follow these rules strictly:\n\n"
                "### Rules:\n"
                "1. **Course Name:** Extract only the name of the course (e.g., 'Operating System', 'Design and Analysis of Algorithms' etc). "
                "Do not include any numbers, special characters, or course codes. If not found, return NOT FOUND.\n"
                "2. **Slot:** Extract the examination slot (e.g., A1, B2) if available. If no slot is explicitly mentioned, return NOT FOUND.\n"
                "3. **Course Code:** Extract the alphanumeric course code (e.g., CSE2005). If not present, return NOT FOUND.\n"
                "4. **Exam Type:** Identify the type of exam as one of the following: 'Final Assessment Test', "
                "'Continuous Assessment Test - 1', 'Continuous Assessment Test - 2'. If not present, return NOT FOUND.\n\n"
                "Learn from these examples how to extract the required data from the text."
                "Example 1:\n"
                "Input:\n"
                """Final Assessment Test-Nover  

                Course: CSE2005  

                Operating System  

                Class NBR(s):5658/5660/5661/566Z/3007   
                5673/5675/5677/6390/6402/6423/6957  

                Time: Three Hours  

                PART-A(8X5=40 Marks) Answer ALL Questions  

                Operating System is  

                (i) Firmware (i) Software (ii) Hardware (iv) Middleware (M)All List out the various services of the Os and brief about each.  

                Discuss in detail about the following with suitable sketch """
                "Output: \n"
                "{\n"
                "  'course': 'Operating System',\n"
                "  'slot': 'NOT FOUND',\n"
                "  'course-code': 'CSE2005',\n"
                "  'exam-type': 'Final Assessment Test'\n"
                "}\n\n"
                "Example 2:\n"
                "Input :\n"
                """
                Course Code and Course Name :CBs3002-Information Security ProgrammeName&Branch ：B.Tech-BBS :Dr.K.VimalaDevi   
                Faculty Name(s)：VL2024250103232  
                Class Number(s) :17.10.2024   
                Date ofExamination :90minutes Maximum Marks:5   
                Exam Duration  

                # Generalinstruction[s):  

                # Answer All Questions  

                ![](images/4782b95dff91ef8f481792291df5afa9b73f9a8e288baa4cdc3f060f4de985ac.jpg)  

                # SCHOOL OFCOMPUTERSCIENCEAND ENGINEERING  

                SLOT:E1  

                # CONTINUOUSASSESSMENTTEST-II FALLSEMESTER2024-2025  

                ![](images/52d6627fa9ce6d8fdf67eb125405a75b9236693147d1e9532c980fa3b23e5bdc.jpg)  
                """
                "Output: \n"
                "{\n"
                "  'course': 'Information Security',\n"
                "  'slot': 'E1',\n"
                "  'course-code': 'CBS3002',\n"
                "  'exam-type': 'Continuous Assessment Test - 2'\n"
                "}\n\n"
                "If information is missing, return the message as NOT FOUND.\n\n"
                "Return NOT FOUND if you are not sure about any information"
                "Now process the input and extract the fields accordingly.",
        },
        {"role": "user", "content": f"{user_response}"},
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "ExamDetail",
                "parameters": {
                    "type": "object",
                    "title": "ExamDetail",
                    "properties": {
                        "course-name" : {
                            "title" : "course-name",
                            "type" : "string",
                            "description" : "The full name of the course without numbers or special characters. Should only include the subject's title (e.g., 'Operating System', 'Compiler Design', 'Design and Analysis of Algorithms')."
                        },
                        "slot" : {
                            "title": "slot",
                            "type" : "string",
                            "description" : "The slot of the examination. This would be a character followed by a number like A1, A2, B1, F2 etc."
                        },
                        "course-code" : {
                            "title" : "course-code",
                            "type" : "string",
                            "description" : "The course code of the course of the examination. It would be fromatted like some alphabets representing deparment like BCE or CBS and numbers indicating course.",
                        },
                        "exam-type" : {
                            "title" : "exam-type",
                            "type" : "string",
                            "description" : "Final Assessment Test|Continuous Assessment Test - 1|Continuous Assessment Test - 2",
                        },
                    },
                    "required": [
                        "course-name",
                        "slot",
                        "course-code",
                        "exam-type"
                    ],
                },
            },
        }
    ],
    tool_choice={"type": "function", "function": {"name": "ExamDetail"}},
    temperature=0.1,
)

parsed_response = json.loads(response["choices"][0]['message']["function_call"]["arguments"])
print(parsed_response)
with open(f'output-jsons/{files[0]}.json', 'w') as f:
    json.dump(parsed_response, f, indent=4)