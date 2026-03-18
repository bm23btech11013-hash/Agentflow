# # # from openai import OpenAI
# # from dotenv import load_dotenv
# # import os

# # load_dotenv()

# # # print("API Key:", os.getenv("GOOGLE_API_KEY"))

# # # client = OpenAI(
# # #     api_key=os.getenv("GOOGLE_API_KEY"),
# # #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# # # )


# # # response = client.chat.completions.create(
# # #     model="gemini-2.5-flash",
# # #     messages=[
# # #         {"role": "system", "content": "You are a helpful assistant."},
# # #         {"role": "user", "content": "Explain to me how AI works"},
# # #     ],
# # # )

# # # print(response.choices[0].message)


# # from google import genai
# # from google.genai import types


# # client = genai.Client(
# #     api_key=os.getenv("GOOGLE_API_KEY"),
# # )


# # def get_weather(location: str) -> str:
# #     return f"The weather in {location} is sunny."


# # tool = types.Tool(
# #     function_declarations=[
# #         {
# #             "name": "get_weather",
# #             "description": "Get the weather for a location",
# #             "parameters": {
# #                 "location": "string",
# #             },
# #             "required": ["location"],
# #         }
# #     ]
# # )


# # print(tool)


# # response = client.models.generate_content(
# #     model="gemini-2.5-flash",
# #     contents="What is the weather like in Boston?",
# #     config={"generate_content_config": {"tools": [tool]}},
# # )


# from google import genai
# from dotenv import load_dotenv
# from google.genai import types

# load_dotenv()

# client = genai.Client()

# config = types.EmbedContentConfig(output_dimensionality=768)

# res = client.models.embed_content(
#     model="gemini-embedding-001",
#     contents=["Hello world", "How are you?"],
#     config=config,
# )

# print(res.embeddings[0].values)
# print([embedding.values for embedding in res.embeddings])
