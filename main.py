import os
import sys
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

MODEL="claude-sonnet-4-5"

llm = ChatAnthropic(model=MODEL, max_tokens=1024, api_key=os.environ.get("ANTHROPIC_API_KEY"))

def ask(message:str, history:list)->str:
    messages = (history or []) + [HumanMessage(content=message)]
    response = llm.invoke(messages)
    return response.content


def chat_loop():
    print(f"Chatting with {MODEL}. Type 'exit' or Ctrl+C to quit.\n")
    history: list = []
    while True:
        try:
            user_input= input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("Bye!!")
            
        if not user_input:
            continue
        if user_input.lower() in {"exit","quit"}:
            print("Bye!!")
            break
            
        history.append(HumanMessage(content=user_input))
        reply=ask(user_input,history=history[:-1])
        print(f"Claude: {reply}\n")
        history.append(AIMessage(content=reply))

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Error: set the ANTHROPIC_API_KEY environment variable first.")
 
    if len(sys.argv) > 1:
        one_shot(" ".join(sys.argv[1:]))
    else:
        chat_loop()
    