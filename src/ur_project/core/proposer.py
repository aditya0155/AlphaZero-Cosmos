# src/ur_project/core/proposer.py

import random
from typing import Dict, Any, Protocol, List

class Task(Protocol):
    id: str
    description: str
    data: Any  # Task-specific data, e.g., numbers for arithmetic, grid for ARC
    expected_solution_type: str # e.g. "integer", "boolean", "string", "grid"

class SimpleArithmeticTask:
    def __init__(self, task_id: str, num1: int, num2: int, operator: str):
        self.id = task_id
        self.num1 = num1
        self.num2 = num2
        self.operator = operator
        self.description = f"Calculate: {num1} {operator} {num2}"
        self.data = {"num1": num1, "num2": num2, "operator": operator}
        self.expected_solution_type = "float" # Use float to handle division

    def get_correct_answer(self) -> float:
        if self.operator == '+':
            return float(self.num1 + self.num2)
        elif self.operator == '-':
            return float(self.num1 - self.num2)
        elif self.operator == '*':
            return float(self.num1 * self.num2)
        elif self.operator == '/':
            if self.num2 == 0:
                raise ValueError("Division by zero in task generation.")
            return float(self.num1 / self.num2)
        raise ValueError(f"Unknown operator: {self.operator}")

class BaseProposer(Protocol):
    def propose_task(self) -> Task:
        ...

class SimpleArithmeticProposer(BaseProposer):
    """Proposes simple arithmetic tasks (e.g., 2 + 3)."""
    def __init__(self, max_number: int = 100, operators: List[str] = None):
        self.max_number = max_number
        self.operators = operators if operators else ['+', '-', '*', '/']
        self._task_counter = 0

    def propose_task(self) -> SimpleArithmeticTask:
        num1 = random.randint(0, self.max_number)
        num2 = random.randint(0, self.max_number)
        operator = random.choice(self.operators)

        if operator == '/' and num2 == 0:
            num2 = random.randint(1, self.max_number) # Avoid division by zero
        
        self._task_counter += 1
        task_id = f"arithmetic_task_{self._task_counter}"
        return SimpleArithmeticTask(task_id, num1, num2, operator)

# Example Usage:
# if __name__ == "__main__":
#     proposer = SimpleArithmeticProposer(max_number=10)
#     for _ in range(5):
#         task = proposer.propose_task()
#         print(f"Proposed Task ID: {task.id}")
#         print(f"Description: {task.description}")
#         print(f"Data: {task.data}")
#         print(f"Expected Solution Type: {task.expected_solution_type}")
#         try:
#             print(f"Correct Answer: {task.get_correct_answer()}")
#         except ValueError as e:
#             print(f"Error getting answer: {e}")
#         print("---") 