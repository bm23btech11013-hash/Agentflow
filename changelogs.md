Version 0.6.3 later
1. Introduced ToolResult for Tool Result
this way you can update state and also return the result of the tool execution in one step, and it will be used in the next step of the agent execution


Version: 0.6.1 later
1. Fix Bugs
2. Reasoning Configuration for Agents



Version: 0.5.7 later
1. Fix Bugs
2. Improved Agent class Now supports tool filtering by tags
3. Added @tool decorator for function-based tools you can add tags, description, that tag can be used
for filtering
4. Added Unit Testing For Agents Introduce `testing` module
5. Added Evaluation Framework for Agents and `evaluation` module
6. Added Google SDK support for converters


Version: 0.5.6 later
2. Improved Error Handling in PgCheckpointer Methods and if thread name is null during update
then default to existing name and it will return is created or not

Version: 0.5.4 - 0.5.5
1. Fix minor bugs in MCP and Node execution with interrupt handling

Version: 0.5.3 later
1. Enhanced MCP Tool Execution to Include User Information in Input Data
2. Created AgentClass for Simplified Agent Instantiation

Version: 0.5.2 later
1. Added Tag-based Tool Filtering for MCP Tools and Local Tools
2. Added Tag Decorator for Function Tools


Features
1. Added Handoff Tools
2. Added Remote Tools


Fixes
1. Parallel tool calls for invoke methods
2. Fixed ID generation for 'int' and 'bigint' types
3. Updated logging to use specific logger names


Version: 0.5.1 later
1. Fixed timestamp handling in PgCheckpointer to avoid validation errors
2. Added updated_at field when reconstructing threads in PgCheckpointer

