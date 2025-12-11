---
date created: 2025-12-11
date updated: 2025-12-11 17:56 UTC
---

# Setting Up Neo4j for Graphiti Memory

This guide shows how to set up Neo4j for persistent memory in Squadron.

> **Note**: Neo4j is **optional**. Squadron works without it using in-memory storage.
> Only set up Neo4j if you need memory persistence across agent restarts.

## Quick Start Options

### Option 1: Docker (Recommended)

The fastest way to get Neo4j running locally:

```bash
# Pull and run Neo4j
docker run -d \
  --name squadron-neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/squadron123 \
  -v neo4j_data:/data \
  neo4j:5-community

# Verify it's running
docker ps | grep neo4j
```

Then configure Squadron:

```bash
# Add to your .env file
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=squadron123
```

**Access the Neo4j Browser**: http://localhost:7474

### Option 2: Neo4j Desktop (GUI)

For a visual interface:

1. Download [Neo4j Desktop](https://neo4j.com/download/)
2. Install and launch
3. Create a new project
4. Add a local database (Neo4j 5.x)
5. Set password and start the database
6. Note the bolt URL (usually `bolt://localhost:7687`)

### Option 3: Neo4j AuraDB (Cloud)

For production or if you don't want to manage infrastructure:

1. Go to [Neo4j AuraDB](https://neo4j.com/cloud/aura/)
2. Create a free account
3. Create a new database (Free tier available)
4. Copy the connection details:

```bash
# .env
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-generated-password>
```

### Option 4: Docker Compose (With Squadron)

For a complete development setup:

```yaml
# docker-compose.yml
version: '3.8'

services:
  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/squadron123
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  neo4j_data:
  neo4j_logs:
```

Run with:

```bash
docker-compose up -d
```

## Verifying the Connection

Test that Squadron can connect:

```python
import asyncio
from squadron.memory import GraphitiMemory

async def test_connection():
    memory = GraphitiMemory()
    await memory.initialize()
    
    if memory._graphiti_client:
        print("✅ Connected to Neo4j!")
    else:
        print("⚠️ Using in-memory fallback (Neo4j not connected)")

asyncio.run(test_connection())
```

## Configuration Reference

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `NEO4J_URI` | Bolt connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Database username | `neo4j` |
| `NEO4J_PASSWORD` | Database password | (required) |

## Graphiti Schema

Squadron/Graphiti creates these node types:

```
(:Entity {id, name, type, created_at, updated_at})
(:Fact {id, content, source, valid_from, valid_to})

Relationships:
(Entity)-[:RELATES_TO {type, weight}]->(Entity)
(Entity)-[:HAS_FACT]->(Fact)
```

## Common Issues

### "Connection refused"

```
Neo4j is not running or wrong port.

Fix:
- Check Docker: docker ps | grep neo4j
- Verify port 7687 is open
- Check NEO4J_URI in .env
```

### "Authentication failed"

```
Wrong username or password.

Fix:
- Verify NEO4J_USER and NEO4J_PASSWORD
- Reset password in Neo4j Browser (http://localhost:7474)
```

### "Database not found"

```
For AuraDB, ensure you're using the correct database name.

Fix:
- Check the connection string from AuraDB console
- Use neo4j+s:// (not bolt://) for cloud
```

## Performance Tips

1. **Add indexes** for faster queries:
   ```cypher
   CREATE INDEX entity_name FOR (e:Entity) ON (e.name);
   CREATE INDEX fact_valid FOR (f:Fact) ON (f.valid_from, f.valid_to);
   ```

2. **Use connection pooling** for production:
   ```python
   # In MemoryConfig
   config = MemoryConfig(
       neo4j_max_connections=50,
       neo4j_connection_timeout=30,
   )
   ```

3. **Regular cleanup** of old facts:
   ```cypher
   MATCH (f:Fact) 
   WHERE f.valid_to < datetime() - duration('P30D')
   DELETE f
   ```

## Without Neo4j

If you don't need persistence, Squadron works fine without Neo4j:

```python
# This just works - uses in-memory storage
from squadron.memory import GraphitiMemory

memory = GraphitiMemory()  # No Neo4j? No problem.
await memory.store(messages)
result = await memory.retrieve("query")
```

The only difference:
- Memory is lost when the process exits
- Fine for demos, testing, and stateless agents
