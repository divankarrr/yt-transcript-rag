from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
ytt=YouTubeTranscriptApi()
transcript=ytt.fetch(video_id='dQw4w9WgXcQ',languages=['en'])  
# print(transcript)
subtitles = []
for snippet in transcript:
    subtitles.append(snippet.text)
final_transcript = " ".join(subtitles)
# print(final_transcript)

splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
listofdoc=splitter.create_documents([final_transcript])
# print(listofdoc)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents=listofdoc, embedding=embeddings)
retriever = vectorstore.as_retriever()
question = "What is the main topic of the video?"
results = retriever.invoke(question)
# print(results)
reslist = []
for doc in results:
    reslist.append(doc.page_content)
AugmentedText = "\n\n".join(reslist)
# print(AugmentedText)


llm_obj=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",task="text-generation")
model_name=ChatHuggingFace(llm=llm_obj)
template=PromptTemplate(template="""

You are a helpful assistant that answers questions based on the transcript context .
Very important: If the answer is not contained within the context below, say "I don't know" and do not try to make up an answer.
{context}
Question: {question}
""",input_variables=["context","question"])
prompt=template.invoke({"context":AugmentedText,"question":question})
result=model_name.invoke(prompt)
print(result.content)

