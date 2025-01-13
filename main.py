from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from hardtack.agent import get_bot_response

app = FastAPI()

@app.post("/bot-response/")
async def bot_response(payload: Dict[str, Any]):
    """
    Endpoint to hardtack.get_bot_response.
    Arg:
        payload (dict): required: "message", optional: "model", "temp", etc.

    Returns:
        Full chatbot response as a single response.
    """
    try:
        # Extract parameters from the payload
        message = payload.get("message")
        chat_history = payload.get("chat_history", [])
        most_recent_query = payload.get("most_recent_query", "")
        selected_recipe = payload.get("selected_recipe", {})
        model = payload.get("model", "openai")
        temp = payload.get("temp", 0.6)

        if not message:
            raise HTTPException(status_code=400, detail="The 'message' field is required.")

        # get response from get_bot_response
        response = get_bot_response(
            message=message,
            chat_history=chat_history,
            most_recent_query=most_recent_query,
            selected_recipe=selected_recipe,
            model=model,
            temp=temp
        )

        # ensure a proper return type
        if isinstance(response, str):
            return {"response": response}
        else:
            raise HTTPException(status_code=500, detail="Invalid response type from get_bot_response.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
