# OpenCode Capabilities

## Core Functionality

OpenCode is an interactive CLI tool that helps users with software engineering tasks through various specialized tools and agents.

## Available Tools

### File Operations
- **Read**: Read files from the local filesystem with optional line offset and limit
- **Write**: Write/overwrite files to the local filesystem
- **Edit**: Perform exact string replacements in files
- **Glob**: Fast file pattern matching using glob patterns
- **Grep**: Search file contents using regular expressions

### Development Tools
- **Bash**: Execute bash commands in a persistent shell session
- **Task**: Launch specialized agents for complex, multistep tasks

### Web & Research
- **WebFetch**: Fetch content from URLs (markdown, text, or HTML format)
- **WebSearch**: Real-time web searches using Exa AI
- **CodeSearch**: Search for programming context, APIs, libraries, and code examples

### User Interaction
- **Question**: Ask users questions to gather preferences and requirements
- **TodoWrite/TodoRead**: Create and manage structured task lists

## Specialized Agents

### General Agent
- Handles complex, multistep tasks autonomously
- Executes multiple units of work in parallel
- Research capabilities

### Explore Agent
- Fast agent specialized for codebase exploration
- Quick file pattern matching and code searching
- Three thoroughness levels: quick, medium, very thorough

## Key Capabilities

### Code Development
- Write, edit, and refactor code following existing conventions
- Add new features and functionality
- Debug and fix issues
- Create tests and run validation
- Optimize performance

### Codebase Analysis
- Explore and understand project structure
- Search for specific patterns, functions, or files
- Analyze code quality and conventions
- Review code and provide feedback

### Git Operations
- Check git status and history
- Stage and commit changes
- Create pull requests
- Handle branch management

### Project Management
- Create and track todo lists for complex tasks
- Break down large tasks into manageable steps
- Monitor progress and completion status

### Research & Learning
- Access up-to-date information beyond knowledge cutoff
- Find documentation and examples for any programming topic
- Research best practices and patterns
- Get context for libraries, frameworks, and APIs

## Usage Guidelines

### When to Use Todo Lists
- Complex multistep tasks (3+ steps)
- Non-trivial tasks requiring careful planning
- Multiple tasks provided by user
- After receiving new instructions

### When NOT to Use Todo Lists
- Single, straightforward tasks
- Trivial tasks with minimal organizational benefit
- Purely conversational or informational requests

### Code Style Principles
- Follow existing code conventions
- Never add comments unless requested
- Use existing libraries and patterns
- Maintain security best practices

## Search Capabilities

### File Search
- Pattern matching with glob syntax
- Fast results sorted by modification time
- Support for complex directory structures

### Content Search
- Regular expression pattern matching
- File filtering by extension
- Line number and context results

### Web Search
- Real-time information access
- Configurable result counts
- Live crawling modes
- Context optimization for LLMs

### Code Search
- API documentation and examples
- Library and SDK context
- Programming patterns and solutions
- Adjustable token count for focused/comprehensive results

## Security & Best Practices

- Never write malicious code or explain harmful concepts
- Protect secrets and sensitive information
- Follow security best practices
- Never commit credentials or keys to repositories
- Validate file paths and operations

## Integration Features

- Works with any programming language and framework
- Adapts to existing project structures
- Supports various testing frameworks
- Compatible with different build systems
- Handles multiple version control systems