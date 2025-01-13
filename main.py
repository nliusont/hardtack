from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any
from hardtack.agent import get_bot_response

app = FastAPI()

@app.post("/bot-response/")
async def bot_response(payload: Dict[str, Any]):
    """
    Endpoint to hardtack.get_bot_response
    Arg:
        payload (dict): required: "message", optional: "model", "temp", etc.

    Returns:
        Streamed chatbot response.
    """
    try:
        message = payload.get("message")
        most_recent_query = payload.get("most_recent_query")
        selected_recipe = payload.get('selected_recipe')
        model = payload.get("model", "openai")
        temp = payload.get("temp", 0.6)

        if not message:
            raise HTTPException(status_code=400, detail="The 'message' field is required.")

        # wrap the `get_bot_response` generator as a streaming response
        def response_generator():
            for chunk in get_bot_response(
                message, 
                most_recent_query=most_recent_query,
                selected_recipe=selected_recipe,
                model=model, 
                temp=temp
                ):
                yield chunk

        return StreamingResponse(response_generator(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
