import  json
from llama_cpp import Llama

llm = Llama(
    model_path="C:/Users/shour/.cache/lm-studio/models/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf", chat_format="chatml-function-calling", n_ctx=2048
)

user_response = """
Final Assessment Test-Nover  

Course: CSE2005  

Operating System  

Class NBR(s):5658/5660/5661/566Z/3007   
5673/5675/5677/6390/6402/6423/6957  

Time: Three Hours  

PART-A(8X5=40 Marks) Answer ALL Questions  

Operating System is  

(i) Firmware (i) Software (ii) Hardware (iv) Middleware (M)All List out the various services of the Os and brief about each.  

Discuss in detail about the following with suitable sketch 
"""

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a data extraction assistant. Your task is to extract specific fields from a given text. "
                "Follow these rules strictly:\n\n"
                "### Rules:\n"
                "1. **Course Name:** Extract only the name of the course (e.g., 'Operating System'). "
                "Do not include any numbers, special characters, or course codes. If not found, return an empty string.\n"
                "2. **Slot:** Extract the examination slot (e.g., A1, B2) if available. If no slot is explicitly mentioned, return an empty string.\n"
                "3. **Course Code:** Extract the alphanumeric course code (e.g., CSE2005). If not present, return an empty string.\n"
                "4. **Exam Type:** Identify the type of exam as one of the following: 'Final Assessment Test', "
                "'Continuous Assessment Test - 1', 'Continuous Assessment Test - 2'. If not present, return an empty string.\n\n"
                "### Examples:\n"
                "#### Input:\n"
                "Course: CSE3001\n"
                "Data Structures and Algorithms\n"
                "Final Assessment Test\n\n"
                "Class Numbers : 5096"
                "#### Output:\n"
                "{\n"
                "  'course': 'Data Structures and Algorithms',\n"
                "  'slot': '',\n"
                "  'course-code': 'CSE3001',\n"
                "  'exam-type': 'Final Assessment Test'\n"
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
                            "description" : "The slot of the examination. This would be a character followed by a number like A1, A2, B1, F2 etc. Leave empty if information is not avaiblable."
                        },
                        "course-code" : {
                            "title" : "course-code",
                            "type" : "string",
                            "description" : "The course code of the course of the examination. It would be fromatted like some alphabets representing deparment like BCE or CBS and numbers indicating course. Leave empty if information is not available.",
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
#with open('user_details_5.json', 'w') as f:
#    json.dump(parsed_response, f, indent=4)