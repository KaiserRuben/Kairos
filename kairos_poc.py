"""
Smart Calendar POC with Project Context Awareness and LLM Integration

This POC demonstrates intelligent calendar management with project context awareness using LLMs.
It focuses on:
- Natural language understanding of scheduling constraints
- Pattern recognition in scheduling behavior
- Intelligent conflict resolution
- Automated schedule optimization
- Project context awareness and continuity

Features:
1. Project Context Management
   - Tracks project progress and status
   - Maintains history of work sessions
   - Manages next steps and priorities

2. Calendar Intelligence
   - Smart conflict detection
   - Pattern-based scheduling
   - Work continuity optimization
   - Break time management

3. LLM Integration
   - Schedule analysis
   - Pattern recognition
   - Intelligent scheduling suggestions
   - Natural language understanding

Extension Points (marked with EXTENSION comments throughout the code):
1. Data Storage
   - Database integration for persistence
   - Multiple calendar support
   - Historical data analysis
   - Cloud sync capabilities

2. Advanced Scheduling
   - Team availability integration
   - Location-based scheduling
   - Travel time consideration
   - Energy level optimization
   - Meeting preparation time

3. Project Intelligence
   - Dependency tracking
   - Resource allocation
   - Progress prediction
   - Risk assessment
   - Automated status updates

4. Calendar Analysis
   - productivity metrics
   - Pattern recognition
   - Scheduling optimization
   - Work-life balance tracking

5. Collaboration
   - Team scheduling
   - Resource sharing
   - Conflict resolution
   - Availability management

Requirements:
- requests
- python-dateutil
- click (for CLI)
"""

import json
import click
import requests
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of calendar events

    EXTENSION: Add more specific event types:
    - External meetings vs internal
    - Different focus categories
    - Training/learning time
    - Customer interactions
    """
    MEETING = "meeting"
    FOCUS = "focus"
    ADMIN = "admin"
    BREAK = "break"


class ProjectPhase(Enum):
    """Project lifecycle phases

    EXTENSION: 
    - Add sub-phases
    - Add phase transition rules
    - Add phase-specific metrics
    - Add phase dependencies
    """
    PLANNING = "planning"
    DEVELOPMENT = "development"
    REVIEW = "review"
    MAINTENANCE = "maintenance"


@dataclass
class ProjectContext:
    """Tracks project context and progress

    Attributes:
        name: Project identifier
        last_session: Timestamp of last work session
        total_time_spent: Total minutes spent on project
        phase: Current project phase
        status: Current status description
        next_steps: List of upcoming tasks
        estimated_time_needed: Estimated minutes to completion

    EXTENSION:
    - Add team members and roles
    - Add dependencies on other projects
    - Add progress metrics
    - Add resource requirements
    - Add risk assessments
    """
    name: str
    last_session: datetime
    total_time_spent: int
    phase: ProjectPhase
    status: str
    next_steps: List[str]
    estimated_time_needed: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "last_session": self.last_session.isoformat(),
            "total_time_spent": self.total_time_spent,
            "phase": self.phase.value,
            "status": self.status,
            "next_steps": self.next_steps,
            "estimated_time_needed": self.estimated_time_needed
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ProjectContext':
        """Create from dictionary"""
        return cls(
            name=data["name"],
            last_session=datetime.fromisoformat(data["last_session"]),
            total_time_spent=data["total_time_spent"],
            phase=ProjectPhase(data["phase"]),
            status=data["status"],
            next_steps=data["next_steps"],
            estimated_time_needed=data["estimated_time_needed"]
        )


@dataclass
class Event:
    """Calendar event with project context

    Attributes:
        title: Event name
        start_time: Start timestamp
        end_time: End timestamp
        event_type: Type of event
        project: Associated project (if any)
        may_be_moved: Whether event can be rescheduled
        recurring: Whether event repeats
        context: Additional event metadata

    EXTENSION:
    - Add attendees and roles
    - Add location (physical/virtual)
    - Add prerequisites
    - Add outcomes/deliverables
    - Add priority levels
    """
    title: str
    start_time: datetime
    end_time: datetime
    event_type: EventType
    project: Optional[str] = None
    may_be_moved: bool = True
    recurring: bool = False
    context: Dict = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "title": self.title,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "event_type": self.event_type.value,
            "project": self.project,
            "may_be_moved": self.may_be_moved,
            "recurring": self.recurring,
            "context": self.context or {}
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Event':
        """Create from dictionary"""
        return cls(
            title=data["title"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            event_type=EventType(data["event_type"]),
            project=data.get("project"),
            may_be_moved=data.get("may_be_moved", True),
            recurring=data.get("recurring", False),
            context=data.get("context", {})
        )


class ProjectManager:
    """Manages project contexts and tracks progress

    EXTENSION:
    - Add persistence layer
    - Add project templates
    - Add progress tracking
    - Add resource management
    - Add cross-project dependencies
    """

    def __init__(self):
        """Initialize with empty project dictionary"""
        self.projects: Dict[str, ProjectContext] = {}
        self.add_sample_projects()

    def add_sample_projects(self):
        """Add sample projects for testing/demo

        Creates three realistic projects with varying states,
        phases, and next steps.
        """
        projects = [
            ProjectContext(
                name="Mobile App Redesign",
                last_session=datetime.now() - timedelta(hours=16),
                total_time_spent=1840,  # About 30 hours
                phase=ProjectPhase.DEVELOPMENT,
                status="Sprint 2/3 - Implementing Core Features",
                next_steps=[
                    "Implement dark mode across all screens",
                    "Fix navigation performance issues",
                    "Complete user settings page",
                    "Update animation transitions",
                    "Integrate with new API endpoints"
                ],
                estimated_time_needed=2400  # 40 hours remaining
            ),
            ProjectContext(
                name="Database Migration",
                last_session=datetime.now() - timedelta(days=2),
                total_time_spent=960,  # 16 hours
                phase=ProjectPhase.PLANNING,
                status="Finalizing Migration Strategy",
                next_steps=[
                    "Complete data mapping documentation",
                    "Set up staging environment",
                    "Create rollback procedures",
                    "Schedule maintenance window",
                    "Update monitoring dashboards"
                ],
                estimated_time_needed=3600  # 60 hours needed
            ),
            ProjectContext(
                name="Security Audit",
                last_session=datetime.now() - timedelta(hours=20),
                total_time_spent=480,  # 8 hours
                phase=ProjectPhase.REVIEW,
                status="Addressing Critical Findings",
                next_steps=[
                    "Patch identified vulnerabilities",
                    "Update dependency versions",
                    "Review access control policies",
                    "Document security improvements",
                    "Prepare executive summary"
                ],
                estimated_time_needed=1200  # 20 hours needed
            )
        ]

        for project in projects:
            self.projects[project.name] = project

    def add_project(self, project: ProjectContext) -> None:
        """Add a new project to the manager"""
        self.projects[project.name] = project

    def update_project_time(self, project_name: str, duration: int) -> None:
        """Update project time spent and last session"""
        if project_name in self.projects:
            project = self.projects[project_name]
            project.total_time_spent += duration
            project.last_session = datetime.now()

    def get_next_steps(self, project_name: str) -> List[str]:
        """Get next steps for a project"""
        return self.projects[project_name].next_steps if project_name in self.projects else []


class CalendarManager:
    """Manages calendar operations and conflict detection

    EXTENSION:
    - Add calendar sync (Google, Outlook)
    - Add recurring event patterns
    - Add timezone support
    - Add team availability
    - Add location constraints
    """

    def __init__(self):
        """Initialize with empty event list"""
        self.events: List[Event] = []
        self.add_sample_events()

    def add_sample_events(self):
        """Add sample events for testing/demo

        Creates a realistic weekly schedule with:
        - Daily standups and admin time
        - Project meetings
        - Focus time blocks
        - Regular breaks
        """
        now = datetime.now()
        sample_events = []

        # Create events for the next 5 days
        for day_offset in range(5):
            day = now + timedelta(days=day_offset)

            # Skip weekends
            if day.weekday() >= 5:
                continue

            # Daily recurring events
            sample_events.extend([
                Event(
                    title="Team Standup",
                    start_time=day.replace(hour=9, minute=30),
                    end_time=day.replace(hour=9, minute=45),
                    event_type=EventType.MEETING,
                    recurring=True,
                    may_be_moved=False,
                    context={"team": "Engineering", "link": "meet.google.com/abc-123"}
                ),
                Event(
                    title="Email & Planning",
                    start_time=day.replace(hour=9, minute=0),
                    end_time=day.replace(hour=9, minute=30),
                    event_type=EventType.ADMIN,
                    recurring=True,
                    context={"priority": "medium"}
                )
            ])

            # Project-specific events
            if day.weekday() == 0:  # Monday
                sample_events.extend([
                    Event(
                        title="Mobile App Sprint Planning",
                        start_time=day.replace(hour=11, minute=0),
                        end_time=day.replace(hour=12, minute=0),
                        event_type=EventType.MEETING,
                        project="Mobile App Redesign",
                        recurring=True,
                        context={"team": ["Engineering", "Design"], "sprint": 2}
                    ),
                    Event(
                        title="Security Review",
                        start_time=day.replace(hour=14, minute=0),
                        end_time=day.replace(hour=15, minute=0),
                        event_type=EventType.MEETING,
                        project="Security Audit",
                        may_be_moved=False,
                        context={"priority": "high", "attendees": ["Security Team", "Tech Lead"]}
                    )
                ])
            elif day.weekday() == 2:  # Wednesday
                sample_events.extend([
                    Event(
                        title="Database Migration Planning",
                        start_time=day.replace(hour=13, minute=0),
                        end_time=day.replace(hour=14, minute=30),
                        event_type=EventType.MEETING,
                        project="Database Migration",
                        context={"team": ["DBA", "Engineering"], "phase": "planning"}
                    ),
                    Event(
                        title="Tech All-Hands",
                        start_time=day.replace(hour=15, minute=0),
                        end_time=day.replace(hour=16, minute=0),
                        event_type=EventType.MEETING,
                        recurring=True,
                        may_be_moved=False,
                        context={"department": "Technology", "link": "meet.google.com/xyz-789"}
                    )
                ])
            elif day.weekday() == 4:  # Friday
                sample_events.extend([
                    Event(
                        title="Mobile App Demo",
                        start_time=day.replace(hour=14, minute=0),
                        end_time=day.replace(hour=15, minute=0),
                        event_type=EventType.MEETING,
                        project="Mobile App Redesign",
                        context={"team": ["Product", "Engineering", "Design"], "sprint": 2}
                    )
                ])

            # Focus time blocks - vary by day
            if day.weekday() in [0, 2, 4]:  # Mon, Wed, Fri
                sample_events.append(
                    Event(
                        title="Mobile App Development",
                        start_time=day.replace(hour=10, minute=0),
                        end_time=day.replace(hour=12, minute=0),
                        event_type=EventType.FOCUS,
                        project="Mobile App Redesign",
                        context={"feature": "dark mode implementation"}
                    )
                )
            if day.weekday() in [1, 3]:  # Tue, Thu
                sample_events.extend([
                    Event(
                        title="Security Implementation",
                        start_time=day.replace(hour=10, minute=0),
                        end_time=day.replace(hour=11, minute=30),
                        event_type=EventType.FOCUS,
                        project="Security Audit",
                        context={"task": "vulnerability patching"}
                    ),
                    Event(
                        title="Database Migration Work",
                        start_time=day.replace(hour=14, minute=0),
                        end_time=day.replace(hour=16, minute=0),
                        event_type=EventType.FOCUS,
                        project="Database Migration",
                        context={"task": "documentation and staging setup"}
                    )
                ])

            # Break times
            sample_events.append(
                Event(
                    title="Lunch Break",
                    start_time=day.replace(hour=12, minute=0),
                    end_time=day.replace(hour=13, minute=0),
                    event_type=EventType.BREAK,
                    recurring=True,
                    context={"type": "lunch"}
                )
            )

        for event in sample_events:
            self.events.append(event)

    def add_event(self, event: Event) -> bool:
        """Add event if no conflicts exist

        Args:
            event: Event to add

        Returns:
            bool: True if added successfully, False if conflicts exist
        """
        if not self._check_conflicts(event):
            self.events.append(event)
            return True
        return False

    def _check_conflicts(self, event: Event) -> List[Event]:
        """Check for conflicting events

        Args:
            event: Event to check for conflicts

        Returns:
            List[Event]: List of conflicting events

        EXTENSION:
        - Add partial overlap detection
        - Add buffer time requirements
        - Add priority-based conflict resolution
        - Add team availability checking
        """
        conflicts = []
        for existing in self.events:
            if (event.start_time < existing.end_time and
                    event.end_time > existing.start_time):
                conflicts.append(existing)
        return conflicts

    def get_free_slots(self, duration: int, start_date: datetime,
                       end_date: datetime) -> List[datetime]:
        """Find free time slots of given duration

        Args:
            duration: Required duration in minutes
            start_date: Search start time
            end_date: Search end time

        Returns:
            List[datetime]: List of potential start times

        EXTENSION:
        - Add preferred time ranges
        - Add energy level optimization
        - Add travel time consideration
        - Add preparation time buffers
        - Add meeting grouping optimization
        """
        free_slots = []
        current = start_date

        # Sort events by start time
        sorted_events = sorted(self.events, key=lambda x: x.start_time)

        for event in sorted_events:
            if (event.start_time - current).total_seconds() / 60 >= duration:
                free_slots.append(current)
            current = max(current, event.end_time)

        if (end_date - current).total_seconds() / 60 >= duration:
            free_slots.append(current)

        return free_slots


class LLMInterface:
    """Manages interactions with the LLM for intelligent scheduling

    EXTENSION:
    - Add conversational context
    - Add learning from feedback
    - Add custom scheduling strategies
    - Add multi-modal inputs
    - Add explanation generation
    """

    def __init__(self, api_url: str, model: str = "llama3.1:latest"):
        """Initialize LLM interface

        Args:
            api_url: Ollama API endpoint
            model: Model identifier
        """
        self.api_url = api_url
        self.model = model
        self.headers = {"Content-Type": "application/json"}

    def suggest_optimal_slots(self, project: ProjectContext,
                              free_slots: List[datetime],
                              preferences: Dict) -> List[Tuple[datetime, str]]:
        """Suggest optimal scheduling slots based on project context and preferences

        Args:
            project: Project context
            free_slots: Available time slots
            preferences: Scheduling preferences

        Returns:
            List[Tuple[datetime, str]]: List of (slot, rationale) pairs
        """
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an intelligent scheduling assistant."},
                    {"role": "user", "content": f"""Suggest optimal slots considering:
    Project: {json.dumps(project.to_dict())}
    Available slots: {json.dumps([slot.isoformat() for slot in free_slots])}
    Preferences: {json.dumps(preferences)}"""}
                ],
                "stream": False,
                "format": {
                    "type": "object",
                    "properties": {
                        "suggestions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "slot": {"type": "string"},
                                    "rationale": {"type": "string"}
                                },
                                "required": ["slot", "rationale"]
                            }
                        }
                    },
                    "required": ["suggestions"]
                }
            }

            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            result = json.loads(response.json()["message"]["content"])

            # Convert back to datetime objects
            return [(datetime.fromisoformat(sugg["slot"]), sugg["rationale"])
                    for sugg in result["suggestions"]]
        except Exception as e:
            logger.error(f"Failed to get optimal slots: {e}")
            return []

    def process_clarification(self, interpretation: dict, response: str,
                              question: str) -> dict:
        """Process user clarification response and update interpretation

        Args:
            interpretation: Current interpretation dict
            response: User's clarification response
            question: Original clarification question

        Returns:
            dict: Updated interpretation
        """
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an intelligent scheduling assistant."},
                    {"role": "user", "content": f"""Update scheduling interpretation based on clarification:
    Current interpretation: {json.dumps(interpretation)}
    Question asked: {question}
    User response: {response}"""}
                ],
                "stream": False,
                "format": {
                    "type": "object",
                    "properties": {
                        "updated_interpretation": {
                            "type": "object",
                            "properties": {
                                "project_name": {"type": "string"},
                                "duration_minutes": {"type": "integer"},
                                "preferences": {
                                    "type": "object",
                                    "properties": {
                                        "preferred_time": {"type": "string"},
                                        "urgency": {"type": "string"},
                                        "needs_focus": {"type": "boolean"}
                                    }
                                },
                                "clarification_needed": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "changes_made": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["updated_interpretation", "changes_made"]
                }
            }

            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            result = json.loads(response.json()["message"]["content"])
            return result["updated_interpretation"]
        except Exception as e:
            logger.error(f"Failed to process clarification: {e}")
            return interpretation

    def interpret_scheduling_request(self, user_input: str, projects: dict) -> dict:
        """Use LLM to interpret natural language scheduling request"""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system",
                     "content": "You are an intelligent scheduling assistant. Extract scheduling parameters from user input."},
                    {"role": "user",
                     "content": f"Currently active projects: {list(projects.keys())}\n\nUser request: {user_input}"}
                ],
                "stream": False,
                "format": {
                    "type": "object",
                    "properties": {
                        "project_name": {"type": "string"},
                        "duration_minutes": {"type": "integer"},
                        "preferences": {
                            "type": "object",
                            "properties": {
                                "preferred_time": {"type": "string",
                                                   "enum": ["morning", "afternoon", "evening", "any"]},
                                "urgency": {"type": "string", "enum": ["high", "medium", "low"]},
                                "needs_focus": {"type": "boolean"},
                            }
                        },
                        "clarification_needed": {"type": "array", "items": {"type": "string"}},
                        "suggested_duration": {"type": "integer"},
                        "rationale": {"type": "string"}
                    },
                    "required": ["project_name", "duration_minutes", "preferences", "clarification_needed"]
                },
                "options": {
                    "temperature": 0
                }
            }

            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            return json.loads(response.json()["message"]["content"])
        except Exception as e:
            logger.error(f"Failed to interpret scheduling request: {e}")
            return None

    def analyze_schedule(self, calendar: List[Event],
                         projects: Dict[str, ProjectContext]) -> dict:
        """Analyze schedule and provide insights

        Args:
            calendar: List of calendar events
            projects: Dictionary of project contexts

        Returns:
            dict: Analysis results and recommendations
        """
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an intelligent scheduling assistant."},
                    {"role": "user", "content": f"""Analyze this schedule and project context:
Calendar: {json.dumps([e.to_dict() for e in calendar])}
Projects: {json.dumps({k: v.to_dict() for k, v in projects.items()})}"""}
                ],
                "stream": False,
                "format": {
                    "type": "object",
                    "properties": {
                        "patterns": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "project_insights": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "optimization_suggestions": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "improvements": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["patterns", "project_insights", "optimization_suggestions", "improvements"]
                },
                "options": {
                    "temperature": 0
                }
            }

            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            result = json.loads(response.json()["message"]["content"])
            return result
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return {
                "patterns": [],
                "project_insights": {},
                "optimization_suggestions": [],
                "improvements": []
            }

    def suggest_schedule(self, project: ProjectContext,
                         free_slots: List[datetime]) -> List[Event]:
        """Suggest optimal scheduling based on project context

        Args:
            project: Project context
            free_slots: Available time slots

        Returns:
            List[Event]: Suggested events
        """
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an intelligent scheduling assistant."},
                    {"role": "user", "content": f"""Suggest optimal scheduling for this project:
Project: {json.dumps(project.to_dict())}
Available slots: {json.dumps([slot.isoformat() for slot in free_slots])}"""}
                ],
                "stream": False,
                "format": {
                    "type": "object",
                    "properties": {
                        "suggested_events": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "start_time": {"type": "string"},
                                    "end_time": {"type": "string"},
                                    "event_type": {"type": "string"},
                                    "project": {"type": "string"},
                                    "may_be_moved": {"type": "boolean"},
                                    "recurring": {"type": "boolean"}
                                },
                                "required": ["title", "start_time", "end_time", "event_type"]
                            }
                        },
                        "rationale": {"type": "string"}
                    },
                    "required": ["suggested_events", "rationale"]
                },
                "options": {
                    "temperature": 0
                }
            }

            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            result = json.loads(response.json()["message"]["content"])
            return [Event.from_dict(e) for e in result.get("suggested_events", [])]
        except Exception as e:
            logger.error(f"Failed to get schedule suggestions: {e}")
            return []


class SmartScheduler:
    """Combines all components for intelligent scheduling

    EXTENSION:
    - Add scheduling strategies
    - Add calendar optimization
    - Add team coordination
    - Add resource management
    - Add automated rescheduling
    """

    def __init__(self, llm_api_url: str):
        """Initialize scheduler components

        Args:
            llm_api_url: Ollama API endpoint
        """
        self.project_manager = ProjectManager()
        self.calendar_manager = CalendarManager()
        self.llm = LLMInterface(llm_api_url)

    def analyze_and_optimize(self) -> dict:
        """Analyze schedule and suggest optimizations

        Returns:
            dict: Analysis results and recommendations
        """
        return self.llm.analyze_schedule(
            self.calendar_manager.events,
            self.project_manager.projects
        )


@click.group()
def cli():
    """Smart Calendar POC CLI"""
    pass


@cli.command()
def analyze():
    """Analyze current schedule and projects"""
    scheduler = SmartScheduler("http://localhost:11434/api")
    try:
        insights = scheduler.analyze_and_optimize()
        click.echo(json.dumps(insights, indent=2))
    except Exception as e:
        click.echo(f"Error: {str(e)}")


@cli.command()
def schedule():
    """Interactive intelligent scheduling"""
    scheduler = SmartScheduler("http://localhost:11434/api")

    # Show current context
    projects = scheduler.project_manager.projects
    if not projects:
        click.echo("No projects available.")
        return

    click.echo("\nCurrent projects:")
    for name, project in projects.items():
        click.echo(f"- {name} ({project.phase.value})")

    # Get natural language input
    user_input = click.prompt(
        "\nWhat would you like to schedule? (e.g. 'I need to work on the mobile app for about 2 hours')",
        type=str
    )

    # Interpret request
    interpretation = scheduler.llm.interpret_scheduling_request(user_input, projects)
    if not interpretation:
        click.echo("Sorry, I couldn't understand your request.")
        return

    # Enhanced clarification loop
    original_questions = interpretation["clarification_needed"]
    while interpretation["clarification_needed"]:
        click.echo("\nI need some clarifications:")
        remaining_questions = interpretation["clarification_needed"].copy()

        for question in remaining_questions:
            response = click.prompt(question, type=str)

            # Update interpretation based on response
            new_interpretation = scheduler.llm.process_clarification(
                interpretation, response, question
            )

            # Show what changed
            if "changes_made" in new_interpretation:
                click.echo("\nUpdated understanding based on your response:")
                for change in new_interpretation["changes_made"]:
                    click.echo(f"- {change}")

            interpretation = new_interpretation

            # Break early if no more clarifications needed
            if not interpretation["clarification_needed"]:
                break

        # Prevent infinite loop if questions can't be resolved
        if interpretation["clarification_needed"] == original_questions:
            click.echo("\nI'm having trouble understanding. Let's start over.")
            return

    # Show final interpretation
    click.echo("\nI understand you want to:")
    click.echo(f"- Work on: {interpretation['project_name']}")
    click.echo(f"- Duration: {interpretation['duration_minutes']} minutes")
    click.echo(f"- Preferences: {json.dumps(interpretation['preferences'], indent=2)}")
    if interpretation.get("rationale"):
        click.echo(f"\nRationale: {interpretation['rationale']}")

    if not click.confirm("\nIs this correct?", default=True):
        click.echo("Let's try again.")
        return

    # Find optimal slots
    project = projects[interpretation['project_name']]
    free_slots = scheduler.calendar_manager.get_free_slots(
        interpretation['duration_minutes'],
        datetime.now(),
        datetime.now() + timedelta(days=7)
    )

    # Filter slots based on preferences
    pref_time = interpretation['preferences']['preferred_time']
    if pref_time == "morning":
        free_slots = [s for s in free_slots if 9 <= s.hour <= 12]
    elif pref_time == "afternoon":
        free_slots = [s for s in free_slots if 13 <= s.hour <= 17]
    elif pref_time == "evening":
        free_slots = [s for s in free_slots if 17 <= s.hour <= 20]

    if not free_slots:
        click.echo("No suitable slots found with current preferences.")
        return

    # Get LLM suggestions for optimal slots
    suggestions = scheduler.llm.suggest_optimal_slots(
        project,
        free_slots[:5],
        interpretation['preferences']
    )

    if not suggestions:
        click.echo("Unable to generate scheduling suggestions.")
        return

    click.echo("\nRecommended slots:")
    for i, (slot, rationale) in enumerate(suggestions, 1):
        end_time = slot + timedelta(minutes=interpretation['duration_minutes'])
        click.echo(f"\n{i}. {slot.strftime('%A, %B %d, %H:%M')} - {end_time.strftime('%H:%M')}")
        click.echo(f"   Why: {rationale}")

    slot_index = click.prompt(
        "\nSelect slot number (0 to cancel)",
        type=click.IntRange(0, len(suggestions)),
        default=1
    )

    if slot_index == 0:
        click.echo("Scheduling cancelled.")
        return

    selected_slot = suggestions[slot_index - 1][0]

    # Create and add event
    event = Event(
        title=f"{project.name} - {project.next_steps[0][:30]}..." if project.next_steps else project.name,
        start_time=selected_slot,
        end_time=selected_slot + timedelta(minutes=interpretation['duration_minutes']),
        event_type=EventType.FOCUS if interpretation['preferences'].get('needs_focus') else EventType.MEETING,
        project=project.name,
        context={
            "task": project.next_steps[0] if project.next_steps else None,
            "urgency": interpretation['preferences'].get('urgency', "medium")
        }
    )

    if scheduler.calendar_manager.add_event(event):
        click.echo("\nEvent scheduled successfully!")
        click.echo(json.dumps(event.to_dict(), indent=2))
    else:
        click.echo("\nFailed to schedule event due to conflicts.")


if __name__ == "__main__":
    cli()
