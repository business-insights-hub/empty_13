"""
AgriBot - Kənd Təsərrüfatı RAG Sistemi
FastAPI Web Interface with Azerbaijani Language Support
"""
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from demo_graph_rag import SimpleGraphRAG
from loguru import logger
import os
import traceback as tb

# Initialize FastAPI app
app = FastAPI(title="AgriBot - Kənd Təsərrüfatı Asistentı")

# Add exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    logger.error(tb.format_exc())
    return PlainTextResponse(
        status_code=500,
        content=f"Error: {str(exc)}\n\n{tb.format_exc()}"
    )

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize Graph RAG system
rag_system = None


@app.get("/ping")
async def ping():
    """Simple health check"""
    return {"status": "ok", "message": "AgriBot is running!"}


def get_rag_system():
    """Get or initialize RAG system"""
    global rag_system
    if rag_system is None:
        rag_system = SimpleGraphRAG()
        logger.success("Graph RAG system initialized")
    return rag_system


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page with search interface"""
    # Use default stats to avoid blocking on database connections
    stats = {
        "neo4j_nodes": 24,
        "neo4j_relationships": 2,
        "pinecone_vectors": 47
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stats": stats
        }
    )


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    """Search agricultural knowledge base"""
    try:
        rag = get_rag_system()

        # Vector search
        logger.info(f"Processing Azerbaijani query: {query}")
        vector_results = rag.query_vector_search(query, top_k=3)

        # Extract keywords for graph search (simple approach)
        keywords = query.split()[:3]  # Take first 3 words
        graph_results = []

        for keyword in keywords:
            if len(keyword) > 3:  # Only search meaningful words
                graph_data = rag.query_graph(keyword)
                graph_results.extend(graph_data)

        # Generate answer using LLM
        answer = rag.answer_question(query)

        # Get statistics
        stats = rag.get_statistics()

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "query": query,
                "answer": answer,
                "vector_results": vector_results,
                "graph_results": graph_results[:5],  # Limit to 5
                "stats": stats
            }
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error": str(e),
                "query": query
            }
        )


@app.get("/stats", response_class=HTMLResponse)
async def statistics(request: Request):
    """Show database statistics"""
    try:
        rag = get_rag_system()
        stats = rag.get_statistics()

        return templates.TemplateResponse(
            "stats.html",
            {
                "request": request,
                "stats": stats
            }
        )

    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error": str(e)
            }
        )


# Mount static files (after route definitions)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    global rag_system
    if rag_system:
        rag_system.close()
        logger.info("RAG system closed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
