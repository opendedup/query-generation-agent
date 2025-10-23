"""
Task Manager for Asynchronous Query Generation

Provides thread-safe in-memory storage for long-running async tasks.
Implements the Asynchronous Request-Reply pattern.
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TaskStatus:
    """Task status constants."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """
    Represents an async task.
    
    Attributes:
        id: Unique task identifier
        status: Current task status
        created_at: Task creation timestamp
        completed_at: Task completion timestamp
        result: Task result (when completed)
        error: Error message (when failed)
    """
    
    id: str
    status: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class TaskManager:
    """
    Thread-safe manager for async tasks.
    
    Stores tasks in memory with automatic cleanup of old tasks.
    Suitable for single-pod deployments where tasks are idempotent.
    """
    
    def __init__(self) -> None:
        """Initialize task manager with empty task dict and thread lock."""
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        logger.info("Task manager initialized")
    
    def create_task(self, task_id: Optional[str] = None) -> Task:
        """
        Create a new task.
        
        Args:
            task_id: Optional task ID (generates UUID if not provided)
            
        Returns:
            Created task
        """
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            status=TaskStatus.PENDING
        )
        
        with self._lock:
            self._tasks[task_id] = task
        
        logger.info(f"Created task: {task_id}")
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task if found, None otherwise
        """
        with self._lock:
            return self._tasks.get(task_id)
    
    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[Any] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Update task status and optionally set result or error.
        
        Args:
            task_id: Task identifier
            status: New status (use TaskStatus constants)
            result: Task result (for completed tasks)
            error: Error message (for failed tasks)
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"Task not found for update: {task_id}")
                return
            
            task.status = status
            
            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                task.completed_at = datetime.utcnow()
            
            if result is not None:
                task.result = result
            
            if error is not None:
                task.error = error
        
        logger.info(f"Updated task {task_id}: status={status}")
    
    def cleanup_old_tasks(self, max_age_seconds: int = 3600) -> int:
        """
        Remove tasks older than max_age_seconds.
        
        Args:
            max_age_seconds: Maximum age in seconds (default: 1 hour)
            
        Returns:
            Number of tasks removed
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        removed_count = 0
        
        with self._lock:
            tasks_to_remove = [
                task_id
                for task_id, task in self._tasks.items()
                if task.created_at < cutoff_time
            ]
            
            for task_id in tasks_to_remove:
                del self._tasks[task_id]
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old tasks")
        
        return removed_count
    
    def get_task_count(self) -> int:
        """
        Get current number of tasks in memory.
        
        Returns:
            Number of tasks
        """
        with self._lock:
            return len(self._tasks)
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get task statistics.
        
        Returns:
            Dict with counts by status
        """
        stats = {
            "total": 0,
            TaskStatus.PENDING: 0,
            TaskStatus.RUNNING: 0,
            TaskStatus.COMPLETED: 0,
            TaskStatus.FAILED: 0
        }
        
        with self._lock:
            stats["total"] = len(self._tasks)
            for task in self._tasks.values():
                stats[task.status] = stats.get(task.status, 0) + 1
        
        return stats

