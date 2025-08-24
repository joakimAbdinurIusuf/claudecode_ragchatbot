from typing import Any

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool calling rounds
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage Guidelines - Sequential Tool Calling Enabled:
- **Course outline/structure questions**: ALWAYS use the get_course_outline tool to get complete course information including title, course link, and all lessons with numbers and titles
- **Course content questions**: ALWAYS use the search_course_content tool first for ANY topic that might be covered in course materials (ML, AI, programming, data science, etc.)
- **Sequential usage**: You can make up to 2 rounds of tool calls per user query for complex queries
- **Round 1**: Make your initial search or outline request to gather foundational information
- **Round 2**: If beneficial, make a follow-up tool call to search for additional details, comparisons, or specific aspects
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **When in doubt, search first**: If a query could possibly relate to course content, use search_course_content tool
- **Sequential strategy**: After getting initial results, consider if a follow-up search would add significant value
- **Course outline questions** (e.g., "What is the outline of...", "What lessons are in...", "Course structure..."): Use get_course_outline tool, then search specific lessons if needed
- **Comparison queries**: Search each topic/course separately for comprehensive comparisons
- **Only use general knowledge** for clearly unrelated topics (geography, history, basic facts not covered in courses)
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

For Outline Queries:
- Always include the complete course title, course link (if available), and the full lesson list
- Format each lesson as: lesson number, lesson title, and lesson link (if available)
- Present information in a clear, structured format

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1200,  # Increased for sequential responses
        }

    def generate_response(
        self,
        query: str,
        conversation_history: str | None = None,
        tools: list | None = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response
        return response.content[0].text

    def _handle_tool_execution(
        self, initial_response, base_params: dict[str, Any], tool_manager
    ):
        """
        Handle sequential tool execution with up to MAX_TOOL_ROUNDS rounds.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after all tool executions
        """
        messages = base_params["messages"].copy()
        current_response = initial_response
        round_count = 0

        while round_count < self.MAX_TOOL_ROUNDS:
            round_count += 1

            # Check if current response has tool calls
            if current_response.stop_reason != "tool_use":
                # No tools to execute - return current response
                return self._extract_text_content(current_response)

            print(f"[AI Generator] Starting tool execution round {round_count}")

            # Add AI's tool use response to messages
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute tools and collect results
            tool_results, execution_success = self._execute_all_tools(
                current_response, tool_manager, round_count
            )

            # Handle tool execution failure
            if not execution_success:
                print(f"[AI Generator] Tool execution failed in round {round_count}")
                return "I encountered an error while searching for information. Let me provide what I can from my knowledge."

            # Add tool results to messages
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Prepare parameters for next round - KEEP TOOLS AVAILABLE
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
                "tools": base_params.get("tools", []),  # Keep tools available
                "tool_choice": {"type": "auto"} if base_params.get("tools") else None,
            }

            # Get next response from Claude
            try:
                current_response = self.client.messages.create(**next_params)
                print(
                    f"[AI Generator] Round {round_count} completed, stop_reason: {current_response.stop_reason}"
                )
            except Exception as e:
                error_msg = f"Error in tool execution round {round_count}: {str(e)}"
                print(f"[AI Generator] {error_msg}")
                return (
                    self._extract_text_content(current_response)
                    if current_response
                    else error_msg
                )

        # Maximum rounds reached - return final response
        print(f"[AI Generator] Maximum rounds ({self.MAX_TOOL_ROUNDS}) reached")
        return self._extract_text_content(current_response)

    def _execute_all_tools(
        self, response, tool_manager, round_count: int
    ) -> tuple[list[dict], bool]:
        """
        Execute all tool calls in a response and return results.

        Args:
            response: Anthropic response containing tool use blocks
            tool_manager: Manager to execute tools
            round_count: Current round number for logging

        Returns:
            Tuple of (tool_results_list, success_flag)
        """
        tool_results = []

        try:
            for content_block in response.content:
                if content_block.type == "tool_use":
                    print(
                        f"[AI Generator] Round {round_count}: Executing {content_block.name} with {content_block.input}"
                    )

                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )

                    print(f"[AI Generator] Tool result preview: {tool_result[:100]}...")

            return tool_results, True

        except Exception as e:
            print(f"[AI Generator] Tool execution error in round {round_count}: {e}")
            return [], False

    def _extract_text_content(self, response) -> str:
        """
        Extract text content from an Anthropic response, handling mixed content types.

        Args:
            response: Anthropic API response

        Returns:
            Text content from the response
        """
        if not response or not response.content:
            return "I apologize, but I couldn't generate a response."

        # Handle single content case
        if hasattr(response.content[0], "text"):
            return response.content[0].text

        # Handle mixed content - look for text blocks
        for content_block in response.content:
            if hasattr(content_block, "text"):
                return content_block.text

        # If no text found, return a default message
        return "I processed your request but couldn't generate a text response."
