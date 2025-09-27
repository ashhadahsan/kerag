"""
SPARQL-based knowledge base implementation for KERAG.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS

from core.interfaces import KnowledgeBaseInterface
from core.types import Entity, Triple, RetrievalResult
from core.exceptions import SPARQLError, EntityNotFoundError
from config.settings import KnowledgeBaseConfig


class SPARQLKnowledgeBase(KnowledgeBaseInterface):
    """SPARQL-based knowledge base implementation."""

    def __init__(
        self, endpoint_url: str, default_graph: Optional[str] = None, timeout: int = 30
    ):
        """
        Initialize SPARQL knowledge base.

        Args:
            endpoint_url: SPARQL endpoint URL
            default_graph: Default graph URI (optional)
            timeout: Query timeout in seconds
        """
        self.endpoint_url = endpoint_url
        self.default_graph = default_graph
        self.timeout = timeout

        # Initialize SPARQL wrapper
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(timeout)

        if default_graph:
            self.sparql.addDefaultGraph(default_graph)

    async def get_entity_info(self, entity_id: str) -> Entity:
        """Get detailed information about an entity."""
        try:
            # Query to get entity information
            query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            
            SELECT DISTINCT ?property ?value ?valueType WHERE {{
                <{entity_id}> ?property ?value .
                OPTIONAL {{ ?value a ?valueType }}
                FILTER (?property != rdf:type || ?value != owl:NamedIndividual)
            }}
            ORDER BY ?property
            """

            results = await self._execute_query(query)

            if not results["results"]["bindings"]:
                raise EntityNotFoundError(f"Entity {entity_id} not found")

            # Extract entity information
            attributes = {}
            neighbors = []

            for binding in results["results"]["bindings"]:
                prop = binding["property"]["value"]
                value = binding["value"]["value"]
                value_type = binding.get("valueType", {}).get("value")

                if value_type and value_type.startswith("http"):
                    # This is an entity neighbor
                    neighbor = Entity(
                        id=value,
                        name=self._extract_name_from_uri(value),
                        type=value_type,
                        attributes={},
                        neighbors=[],
                    )
                    neighbors.append(neighbor)
                else:
                    # This is an attribute
                    attributes[prop] = value

            # Get entity type
            entity_type = await self._get_entity_type(entity_id)

            return Entity(
                id=entity_id,
                name=self._extract_name_from_uri(entity_id),
                type=entity_type,
                attributes=attributes,
                neighbors=neighbors,
            )

        except Exception as e:
            raise SPARQLError(f"Failed to get entity info for {entity_id}: {str(e)}")

    async def get_entity_neighbors(
        self, entity_id: str, relations: List[str] = None, max_hops: int = 1
    ) -> RetrievalResult:
        """Get neighbors of an entity up to max_hops."""
        try:
            # Build relation filter
            relation_filter = ""
            if relations:
                relation_list = " ".join([f"<{rel}>" for rel in relations])
                relation_filter = f"FILTER (?relation IN ({relation_list}))"

            # Query for multi-hop neighbors
            hop_patterns = []
            for hop in range(1, max_hops + 1):
                if hop == 1:
                    pattern = f"<{entity_id}> ?relation ?neighbor{hop}"
                else:
                    prev_neighbor = f"?neighbor{hop-1}"
                    pattern = f"{prev_neighbor} ?relation{hop} ?neighbor{hop}"
                hop_patterns.append(pattern)

            # Combine all hop patterns
            hop_query = " . ".join(hop_patterns)

            query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT DISTINCT ?neighbor ?relation ?hop WHERE {{
                {{
                    {hop_query}
                    {relation_filter}
                }}
                UNION {{
                    <{entity_id}> rdfs:label ?label .
                    BIND (0 as ?hop)
                    BIND (<{entity_id}> as ?neighbor)
                    BIND (rdfs:label as ?relation)
                }}
            }}
            ORDER BY ?hop ?relation
            """

            results = await self._execute_query(query)

            # Process results
            entities = []
            triples = []
            seen_entities = set()

            for binding in results["results"]["bindings"]:
                neighbor_uri = binding["neighbor"]["value"]
                relation = binding["relation"]["value"]
                hop = int(binding["hop"]["value"])

                # Create entity if not seen before
                if neighbor_uri not in seen_entities:
                    try:
                        entity = await self.get_entity_info(neighbor_uri)
                        entities.append(entity)
                        seen_entities.add(neighbor_uri)
                    except EntityNotFoundError:
                        continue

                # Create triple
                triple = Triple(
                    subject=entity_id if hop == 1 else f"?neighbor{hop-1}",
                    predicate=relation,
                    object=neighbor_uri,
                )
                triples.append(triple)

            return RetrievalResult(
                entities=entities,
                triples=triples,
                metadata={
                    "source_entity": entity_id,
                    "max_hops": max_hops,
                    "relations_filter": relations,
                    "total_entities": len(entities),
                    "total_triples": len(triples),
                },
            )

        except Exception as e:
            raise SPARQLError(f"Failed to get neighbors for {entity_id}: {str(e)}")

    async def search_entities(
        self, query: str, entity_type: str = None, limit: int = 10
    ) -> List[Entity]:
        """Search for entities matching a query."""
        try:
            # Build type filter
            type_filter = ""
            if entity_type:
                type_filter = f"OPTIONAL {{ ?entity rdf:type <{entity_type}> }}"

            search_query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            
            SELECT DISTINCT ?entity ?label WHERE {{
                ?entity rdfs:label ?label .
                {type_filter}
                FILTER (
                    CONTAINS(LCASE(?label), LCASE("{query}")) ||
                    CONTAINS(LCASE(str(?entity)), LCASE("{query}"))
                )
            }}
            ORDER BY ?label
            LIMIT {limit}
            """

            results = await self._execute_query(search_query)

            entities = []
            for binding in results["results"]["bindings"]:
                entity_uri = binding["entity"]["value"]
                label = binding["label"]["value"]

                try:
                    entity = await self.get_entity_info(entity_uri)
                    entities.append(entity)
                except EntityNotFoundError:
                    # Create minimal entity if full info not available
                    entity = Entity(
                        id=entity_uri,
                        name=label,
                        type=entity_type or "Unknown",
                        attributes={"label": label},
                        neighbors=[],
                    )
                    entities.append(entity)

            return entities

        except Exception as e:
            raise SPARQLError(f"Failed to search entities: {str(e)}")

    async def get_relations(self, entity_id: str) -> List[str]:
        """Get all relations for an entity."""
        try:
            query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT DISTINCT ?relation WHERE {{
                <{entity_id}> ?relation ?object .
                FILTER (?relation != rdf:type)
            }}
            ORDER BY ?relation
            """

            results = await self._execute_query(query)

            relations = []
            for binding in results["results"]["bindings"]:
                relation = binding["relation"]["value"]
                relations.append(relation)

            return relations

        except Exception as e:
            raise SPARQLError(f"Failed to get relations for {entity_id}: {str(e)}")

    async def execute_query(self, query: str) -> RetrievalResult:
        """Execute a custom query against the knowledge base."""
        try:
            results = await self._execute_query(query)

            # Convert SPARQL results to RetrievalResult
            entities = []
            triples = []
            seen_entities = set()

            # Parse results based on common SPARQL patterns
            for binding in results["results"]["bindings"]:
                # Look for entity patterns (entity, name, type)
                if "entity" in binding and "name" in binding:
                    entity_uri = binding["entity"]["value"]
                    entity_name = binding["name"]["value"]
                    entity_type = binding.get("type", {}).get("value", "Unknown")

                    if entity_uri not in seen_entities:
                        entity = Entity(
                            id=entity_uri,
                            name=entity_name,
                            type=entity_type,
                            attributes={"label": entity_name},
                            neighbors=[],
                        )
                        entities.append(entity)
                        seen_entities.add(entity_uri)

                # Look for triple patterns (subject, predicate, object)
                elif (
                    "subject" in binding
                    and "predicate" in binding
                    and "object" in binding
                ):
                    triple = Triple(
                        subject=binding["subject"]["value"],
                        predicate=binding["predicate"]["value"],
                        object=binding["object"]["value"],
                        confidence=binding.get("confidence", {}).get("value"),
                    )
                    triples.append(triple)

            return RetrievalResult(
                entities=entities,
                triples=triples,
                metadata={
                    "query": query,
                    "result_count": len(results["results"]["bindings"]),
                },
            )

        except Exception as e:
            raise SPARQLError(f"Failed to execute query: {str(e)}")

    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a SPARQL query and return results."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._sync_execute_query, query)
            return result
        except Exception as e:
            raise SPARQLError(f"Query execution failed: {str(e)}")

    def _sync_execute_query(self, query: str) -> Dict[str, Any]:
        """Synchronously execute a SPARQL query."""
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
            return results
        except Exception as e:
            raise SPARQLError(f"SPARQL query failed: {str(e)}")

    async def _get_entity_type(self, entity_id: str) -> str:
        """Get the type of an entity."""
        try:
            query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?type WHERE {{
                <{entity_id}> rdf:type ?type .
                FILTER (?type != <http://www.w3.org/2002/07/owl#NamedIndividual>)
            }}
            LIMIT 1
            """

            results = await self._execute_query(query)

            if results["results"]["bindings"]:
                return results["results"]["bindings"][0]["type"]["value"]
            else:
                return "Unknown"

        except Exception:
            return "Unknown"

    def _extract_name_from_uri(self, uri: str) -> str:
        """Extract a human-readable name from a URI."""
        # Remove namespace and get the local name
        if "#" in uri:
            return uri.split("#")[-1]
        elif "/" in uri:
            return uri.split("/")[-1]
        else:
            return uri
