import json
import asyncio
import logging
from pathlib import Path
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
from mcp import types
from mcp.types import ServerCapabilities, ToolsCapability
from database_utils import SQLServerHelper
from utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-sqlserver")

_script_dir = Path(__file__).parent
config = load_config(str(_script_dir / "config.yaml"))
DB_CONFIGS = {
    "shly-rundata": config.get("database", {}),
    "shly-basedb": config.get("basedatabase", {}),
}


def get_db(db_name: str = "shly-rundata") -> SQLServerHelper:
    cfg = DB_CONFIGS.get(db_name)
    if not cfg:
        raise ValueError(f"Unknown database: {db_name}, available: {list(DB_CONFIGS.keys())}")
    helper = SQLServerHelper(
        server=cfg["server"],
        port=cfg["port"],
        database=cfg["database"],
        username=cfg["username"],
        password=cfg["password"],
    )
    helper.connect()
    return helper


server = Server("sqlserver-agent")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_database",
            description="Execute arbitrary SQL query on the SQL Server database and return results as JSON",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query to execute (e.g., SELECT TOP 10 * FROM dbo.LogData)",
                    },
                    "params": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional parameters for parameterized queries",
                    },
                    "database": {
                        "type": "string",
                        "enum": ["shly-rundata", "shly-basedb"],
                        "description": "Target database, defaults to shly-rundata",
                    },
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="list_tables",
            description="List all tables and views in the specified database",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "enum": ["shly-rundata", "shly-basedb"],
                        "description": "Target database, defaults to shly-rundata",
                    }
                },
            },
        ),
        types.Tool(
            name="get_table_schema",
            description="Get column information (name, type, nullable, etc.) for a specified table",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Table name with schema (e.g., dbo.LogData)",
                    },
                    "database": {
                        "type": "string",
                        "enum": ["shly-rundata", "shly-basedb"],
                        "description": "Target database, defaults to shly-rundata",
                    },
                },
                "required": ["table_name"],
            },
        ),
        types.Tool(
            name="get_table_sample",
            description="Preview the first N rows of a table to understand its data",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Table name with schema (e.g., dbo.LogData)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of rows to return, defaults to 10",
                    },
                    "database": {
                        "type": "string",
                        "enum": ["shly-rundata", "shly-basedb"],
                        "description": "Target database, defaults to shly-rundata",
                    },
                },
                "required": ["table_name"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    args = arguments or {}
    db_name = args.get("database", "shly-rundata")
    db = get_db(db_name)

    try:
        if name == "query_database":
            sql = args["sql"]
            params = args.get("params", [])
            db.cursor.execute(sql, params)
            rows = db.cursor.fetchall()
            cols = [col[0] for col in db.cursor.description] if db.cursor.description else []
            result = json.dumps([dict(zip(cols, row)) for row in rows], default=str, ensure_ascii=False)
            return [types.TextContent(type="text", text=result)]

        elif name == "list_tables":
            db.cursor.execute("""
                SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
                FROM INFORMATION_SCHEMA.TABLES
                ORDER BY TABLE_TYPE, TABLE_SCHEMA, TABLE_NAME
            """)
            tables = []
            views = []
            for row in db.cursor.fetchall():
                full_name = f"{row[0]}.{row[1]}"
                if row[2] == "BASE TABLE":
                    tables.append(full_name)
                else:
                    views.append(full_name)
            result = json.dumps({"tables": tables, "views": views}, ensure_ascii=False)
            return [types.TextContent(type="text", text=result)]

        elif name == "get_table_schema":
            db.cursor.execute("""
                SELECT
                    COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """, (args["table_name"].split(".")[-1],))
            cols = db.cursor.fetchall()
            schema = [
                {
                    "column": row[0],
                    "type": row[1],
                    "nullable": row[2],
                    "max_length": row[3],
                }
                for row in cols
            ]
            result = json.dumps(schema, ensure_ascii=False)
            return [types.TextContent(type="text", text=result)]

        elif name == "get_table_sample":
            limit = args.get("limit", 10)
            table_name = args["table_name"]
            db.cursor.execute(f"SELECT TOP {limit} * FROM {table_name}")
            rows = db.cursor.fetchall()
            cols = [col[0] for col in db.cursor.description] if db.cursor.description else []
            result = json.dumps([dict(zip(cols, row)) for row in rows], default=str, ensure_ascii=False)
            return [types.TextContent(type="text", text=result)]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    finally:
        try:
            db.close_connection()
        except Exception:
            pass


async def main():
    logger.info("Starting MCP SQL Server agent...")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="sqlserver-agent",
                server_version="1.0.0",
                capabilities=ServerCapabilities(
                    tools=ToolsCapability(listChanged=False),
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
