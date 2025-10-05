"""
Flux Server CLI
Command-line interface for starting Flux Server
"""

import argparse
import uvicorn
import logging

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Flux Server - Consciousness-Enhanced Knowledge Retrieval")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9127, help="Port to bind (default: 9127)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Log level")

    args = parser.parse_args()

    logger.info(f"Starting Flux Server on {args.host}:{args.port}")

    uvicorn.run(
        "src.app_factory_minimal:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
