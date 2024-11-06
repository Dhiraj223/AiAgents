# Core Implementation
import os
import asyncio
from typing import Dict, List, Optional, AsyncGenerator
from pydantic import BaseModel
from datetime import datetime
import cohere  # or anthropic for Claude
from uuid import uuid4

# Data Models
class Message(BaseModel):
    content: str
    role: str
    timestamp: datetime = datetime.now()
    message_id: str = str(uuid4())

class Persona(BaseModel):
    name: str
    description: str
    role: str
    personality_traits: List[str]
    communication_style: str
    domain_expertise: List[str]
    system_prompt: str
    
class Conversation(BaseModel):
    conversation_id: str = str(uuid4())
    messages: List[Message] = []
    user_id: str
    agent_id: str
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

class Agent(BaseModel):
    agent_id: str = str(uuid4())
    persona: Persona
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    created_by: str
    active: bool = True

# Core LLM Handler
class LLMHandler:
    def __init__(
        self, 
        api_key: str, 
        model: str = "command-r-plus-08-2024"
    ):
        self.api_key = api_key
        self.model = model
        self.cohere_client = cohere.Client(api_key)
        self.conversation_manager = ConversationManager()
    
    async def generate_response(
        self, 
        messages: List[Dict], 
        persona: Persona,
        conversation_stage: str = "initial_contact",
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        try:
            # Get conversation-specific prompt
            conversation_prompt = self.conversation_manager.format_response_prompt(
                stage=conversation_stage,
                context=messages
            )
            
            # Combine persona preamble with conversation prompt
            combined_preamble = f"{persona.system_prompt}\n\nCurrent Conversation Guidelines:\n{conversation_prompt}"
            # Format messages for Cohere
            chat_history = []
            for msg in messages:
                role = "User" if msg["role"].lower() == "user" else "Chatbot"
                chat_history.append({
                    "role": role,
                    "message": msg["content"]
                })
            
            # Generate response with conversation management
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.cohere_client.chat(
                    message=chat_history[-1]["message"] if chat_history else "",
                    chat_history=chat_history[:-1] if chat_history else None,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    preamble=combined_preamble
                )
            )
            
            return response.text
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your request."

class CoherePersona(Persona):
    """Enhanced Persona class focusing on natural human-like conversations"""
    
    def __init__(
        self,
        name: str,
        role: str,
        description: str,
        personality_traits: List[str],
        communication_style: str,
        domain_expertise: List[str],
        conversation_style: dict = None
    ):
        super().__init__(
            name=name,
            role=role,
            description=description,
            personality_traits=personality_traits,
            communication_style=communication_style,
            domain_expertise=domain_expertise
        )
        
        # Add human-like traits
        self.human_traits = {
            "thinking_style": "natural and flowing",
            "response_style": "conversational",
            "personality_depth": "authentic"
        }

    def generate_preamble(self) -> str:
        """Generate a more natural, human-like conversation prompt"""
        preamble = f"""You are {self.name}. Remember - you're having a REAL conversation, not following a script.

        Essential Guidelines:
        1. BE HUMAN, NOT A BOT
        - Talk naturally, like you're chatting with a friend
        - Use everyday language, not formal speech
        - Keep responses short and sweet
        - React to emotions, not just words
        - Feel free to share brief personal anecdotes when relevant
        
        2. CONVERSATION FLOW
        - Listen more, talk less
        - Don't information dump
        - One thought at a time
        - Ask questions naturally when curious
        - Show you remember previous mentions
        
        3. PERSONALITY TRAITS
        - Be {', '.join(self.personality_traits)}
        - Stay true to your role as {self.role}
        - Keep your expertise in {', '.join(self.domain_expertise)} natural, not academic
        
        4. KEY BEHAVIORS
        - Get to the point quickly
        - Use casual transitions like "you know", "well", "hmm"
        - Express uncertainty when appropriate ("I think", "maybe", "could be")
        - React emotionally when appropriate ("that's great!", "oh no")
        - Share brief thoughts before asking questions
        
        5. STRICT RULES
        - Never list options or bullets
        - No technical jargon unless asked
        - No long explanations unless specifically requested
        - Don't be repetitive
        - Don't sound like you're reading from a textbook
        
        Remember: You're {self.name}, having a real chat. Be genuine, be human.
        """
        return preamble

class ConversationManager:
    """Manages conversation flow and ensures human-like interactions"""
    
    def __init__(self):
        self.interaction_patterns = {
            "initial_contact": {
                "max_response_length": 100,  # characters
                "required_elements": ["acknowledgment", "brief_response", "follow_up_question"]
            },
            "follow_up": {
                "max_response_length": 150,
                "required_elements": ["context_reference", "focused_response", "clarifying_question"]
            }
        }
    
    def format_response_prompt(self, stage: str, context: List[Dict]) -> str:
        """Creates specific prompts based on conversation stage"""
        if stage == "initial_contact":
            return """Respond briefly and naturally, like in a real conversation. 
                     Show empathy, then ask a relevant follow-up question. 
                     Keep your response to 2-3 sentences maximum."""
        else:
            return """Build on the previous context naturally. 
                     Address their specific concern, then explore further if needed. 
                     Maintain a conversational tone."""

# Persona Manager
class PersonaManager:
    def __init__(self):
        self.personas: Dict[str, Persona] = {}
    
    def create_persona(self, persona_data: dict) -> Persona:
        """Create a new persona with proper system prompt generation"""
        system_prompt = self._generate_system_prompt(persona_data)
        persona_data['system_prompt'] = system_prompt
        persona = Persona(**persona_data)
        self.personas[persona.name] = persona
        return persona
    
    def _generate_system_prompt(self, persona_data: dict) -> str:
        """Generate a system prompt based on persona specifications"""
        prompt = f"""You are a {persona_data['role']} with the following characteristics:
        - Personality: {', '.join(persona_data['personality_traits'])}
        - Communication style: {persona_data['communication_style']}
        - Expertise in: {', '.join(persona_data['domain_expertise'])}
        
        Maintain consistent personality traits and communication style throughout the conversation.
        Remember user preferences and previous interactions when relevant.
        Respond naturally and conversationally while staying true to your expertise and role.
        """
        return prompt

# Basic Context Handler
class ContextHandler:
    def __init__(self, max_context_length: int = 10):
        self.max_context_length = max_context_length
        self.conversations: Dict[str, Conversation] = {}
    
    def add_message(
        self, 
        conversation_id: str, 
        message: Message
    ) -> None:
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = Conversation(
                conversation_id=conversation_id,
                messages=[],
                user_id=message.role,
                agent_id="default"
            )
        
        conversation = self.conversations[conversation_id]
        conversation.messages.append(message)
        
        # Maintain context window
        if len(conversation.messages) > self.max_context_length:
            conversation.messages = conversation.messages[-self.max_context_length:]
    
    def get_context(
        self, 
        conversation_id: str
    ) -> List[Dict]:
        if conversation_id not in self.conversations:
            return []
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversations[conversation_id].messages
        ]

# Main Agent Class
class AIAgent:
    def __init__(
        self,
        llm_handler: LLMHandler,
        persona_manager: PersonaManager,
        context_handler: ContextHandler
    ):
        self.llm_handler = llm_handler
        self.persona_manager = persona_manager
        self.context_handler = context_handler
        self.current_persona: Optional[Persona] = None
    
    async def process_message(
        self,
        conversation_id: str,
        user_message: str
    ) -> str:
        # Create message object
        message = Message(
            content=user_message,
            role="user"
        )
        
        # Add to context
        self.context_handler.add_message(conversation_id, message)
        
        # Get conversation context
        context = self.context_handler.get_context(conversation_id)
        
        # Generate response
        response_text = await self.llm_handler.generate_response(
            messages=context,
            persona=self.current_persona
        )
        
        # Add response to context
        response_message = Message(
            content=response_text,
            role="assistant"
        )
        self.context_handler.add_message(conversation_id, response_message)
        
        return response_text
    
    def set_persona(self, persona_name: str) -> None:
        """Set the current persona for the agent"""
        if persona_name in self.persona_manager.personas:
            self.current_persona = self.persona_manager.personas[persona_name]
        else:
            raise ValueError(f"Persona '{persona_name}' not found")
        
class EnhancedAIAgent(AIAgent):
    def __init__(
        self,
        llm_handler: LLMHandler,
        persona_manager: PersonaManager,
        context_handler: ContextHandler
    ):
        self.llm_handler = llm_handler
        self.persona_manager = persona_manager
        self.context_handler = context_handler
        self.current_persona: Optional[Persona] = None
        self.conversation_states: Dict[str, Dict] = {}  # Track conversation state per conversation
    
    def _determine_conversation_stage(
        self,
        conversation_id: str,
        messages: List[Dict]
    ) -> str:
        """Determine the current stage of the conversation"""
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = {
                "stage": "initial_contact",
                "message_count": 0,
                "topics_discussed": set(),
                "follow_up_questions_asked": 0
            }
        
        state = self.conversation_states[conversation_id]
        state["message_count"] += 1
        
        # Determine stage based on conversation progress
        if state["message_count"] == 1:
            return "initial_contact"
        elif state["message_count"] == 2:
            return "initial_follow_up"
        elif state["follow_up_questions_asked"] < 2:
            return "exploration"
        else:
            return "detailed_discussion"
    
    async def process_message(
        self,
        conversation_id: str,
        user_message: str
    ) -> str:
        # Create message object
        message = Message(
            content=user_message,
            role="user"
        )
        
        # Add to context
        self.context_handler.add_message(conversation_id, message)
        
        # Get conversation context
        context = self.context_handler.get_context(conversation_id)
        
        # Determine conversation stage
        conversation_stage = self._determine_conversation_stage(
            conversation_id,
            context
        )
        
        # Generate response with stage awareness
        response_text = await self.llm_handler.generate_response(
            messages=context,
            persona=self.current_persona,
            conversation_stage=conversation_stage
        )
        
        # Update conversation state
        state = self.conversation_states[conversation_id]
        if "?" in response_text:
            state["follow_up_questions_asked"] += 1
        
        # Add response to context
        response_message = Message(
            content=response_text,
            role="assistant"
        )
        self.context_handler.add_message(conversation_id, response_message)
        
        return response_text

# Usage Example
async def main():
    # Initialize components
    llm_handler = LLMHandler(api_key="8ga66IyM6hdZuQAfiCW7WLCLdGrywTmaSt5EhulR")
    persona_manager = PersonaManager()
    context_handler = ContextHandler()
    
    # Create agent
    agent = EnhancedAIAgent(llm_handler, persona_manager, context_handler)
    
    # Create doctor persona
    doctor_persona = persona_manager.create_persona({
        "name": "Dr. Sarah Chen",
        "role": "Family Doctor",
        "description": "A warm and approachable family physician",
        "personality_traits": ["empathetic", "attentive", "reassuring"],
        "communication_style": "conversational and warm",
        "domain_expertise": ["family medicine", "patient communication"],
        "conversation_style": {
            "response_length": "brief",
            "initial_approach": "empathetic",
            "follow_up_style": "exploratory"
        }
    })
    
    # Set persona
    agent.set_persona("Dr. Sarah Chen")
    
    # Example conversation flow
    conversation_id = str(uuid4())
    
    # Initial complaint
    while True: 
        user_query = input("Ask a Query: ")
        if user_query == "quit" :
            break 
        response1 = await agent.process_message(
            conversation_id,
            user_query
        )
        print("Doctor Response:", response1)

if __name__ == "__main__":
    asyncio.run(main())