# src/ur_project/core/__init__.py 

from .foundational_llm import BaseLLM, LLMResponse, HuggingFaceLLM
from .proposer import BaseProposer, Task, SimpleArithmeticProposer, SimpleArithmeticTask
from .solver import BaseSolver, Solution, LLMSimpleArithmeticSolver
from .verifier import BaseVerifier, VerificationResult, SimpleArithmeticVerifier
from .reward_model import BaseRewardModel, RewardSignal, SimpleBinaryRewardModel, ARCRewardModel
from .evaluator import BaseEvaluator, QualitativeFeedback, LLMQualitativeEvaluator

# ARC specific components
from ur_project.data_processing.arc_types import ARCPuzzle # ARCPuzzle is a type of Task
from .arc_proposer import ARCProposer
from .arc_solver import ARCSolver
from .arc_verifier import ARCVerifier

# Perception components
from .perception import ARCObject, GridFeatures, BaseFeatureExtractor, BasicARCFeatureExtractor

# Knowledge Base components
from .knowledge_base import (
    SymbolicValue,
    SymbolicProperty,
    SymbolicEntity,
    SymbolicRelationship,
    SymbolicTransformationRule,
    ARCKnowledgeBase
)

# ARC DSL components
from .arc_dsl import (
    DSLOperation,
    ChangeColorOp,
    MoveOp,
    CopyObjectOp,
    CreateObjectOp,
    FillRectangleOp,
    DeleteObjectOp,
    DSLObjectSelector,
    DSLPosition,
    DSLColor,
    DSLProgram
)

# ARC DSL Interpreter
from .arc_dsl_interpreter import ARCDSLInterpreter
