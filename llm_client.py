import os
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv


load_dotenv()

#Config
PRIMARY_MODEL=os.environ.get("PRIMARY_MODEL")
FALLBACK_MODEL=os.environ.get("FALLBACK_MODEL")
TIMEOUT_SECONDS=float(os.environ.get("TIMEOUT_SECONDS"))
API_KEY=os.environ.get("ANTHROPIC_API_KEY")

FRIENDLY_ERR_MSG="Sorry! Right now something issue. could you pls try again later?"

#Logging
Path("logs").mkdir(exist_ok=True)
logger=logging.getLogger("day_llm")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
 
    file_handler = RotatingFileHandler("logs/app.log", maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
 
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    
#LLM setup timeout + retry + fallback

_primary_llm = ChatAnthropic(model=PRIMARY_MODEL,max_tokens=1024,api_key=API_KEY,timeout=TIMEOUT_SECONDS)

_fallback_llm = ChatAnthropic(model=FALLBACK_MODEL,max_tokens=1024,api_key=API_KEY,timeout=TIMEOUT_SECONDS)


llm = _primary_llm.with_retry(stop_after_attempt=3,wait_exponential_jitter=True).with_fallbacks([_fallback_llm])


def ask(message:list)-> str:
    start = time.time()
    try:
        response = llm.invoke(message)
    except Exception as e:
        duration_ms = round((time.time() - start) * 1000, 1)
        logger.error("LLM call failed after retries and fallback (%.0fms): %s", duration_ms, e)
        return FRIENDLY_ERR_MSG


    duration_ms = round((time.time()-start)*1000,1)
    model_used=(response.response_metadata or {}).get("model","unknown")
    usage= getattr(response,"usage_metadata",None) or {}
    logger.info(
        "LLM call ok (%.0fms) model=%s input_tokens=%s output_tokens=%s",
        duration_ms,
        model_used,  # if this shows the fallback model's name, the primary failed
        usage.get("input_tokens"),
        usage.get("output_tokens"),
    )
    
    return response.content