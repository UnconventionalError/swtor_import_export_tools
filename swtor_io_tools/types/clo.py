# <pep8 compliant>

"""
Data model for SWTOR .clo (cloth physics) files.

Ported from Jedipedia's clo_binary-read.js / clo-info.js, which document the real
binary layout: a header, a fixed-size string table, and five parallel sections
(bones, particles, particle-radius data, edges, colliders, triangles).

Each cloth "bone" doubles as a physics particle (bone index i drives particle i);
"bones" here also includes references to the *real* character skeleton (Pelvis,
Chest1, Head, ...) by name, via the `parent` field, since those bones are not
themselves stored in this file -- only referenced.
"""

from typing import List


class Cloth:

    class Bone:
        __slots__ = (
            'index', 'name', 'parent',
            'boneToParentRot', 'boneToParentTrans',
            'rootToBoneRot', 'rootToBoneTrans',
            'restEdgeDirection',
            'startParticle', 'endParticle',
        )

        index: int
        name: str
        parent: str  # Name of the parent bone. May reference a bone NOT in this file
        # (a real skeleton bone from the matching .gr2, e.g. "Pelvis", "Chest2", "Head").
        boneToParentRot: List[float]  # quaternion (x, y, z, w)
        boneToParentTrans: List[float]  # (x, y, z)
        rootToBoneRot: List[float]  # quaternion (x, y, z, w); inverse-bind rotation
        rootToBoneTrans: List[float]  # (x, y, z); inverse-bind translation
        restEdgeDirection: List[float]  # (x, y, z) unit vector, start particle -> end particle
        startParticle: int  # -1 if this bone drives no particle segment (leaf / non-cloth)
        endParticle: int  # -1 if unset

        def __str__(self):
            # type: () -> str
            return str({slot: getattr(self, slot, None) for slot in self.__slots__})

    class Particle:
        __slots__ = (
            'index', 'drivenBone', 'damping', 'movementForceFactor',
            'invertedSimWeight', 'isZeroWeight', 'isOneWeight',
            'colliderBitflag', 'radius',
        )

        index: int
        drivenBone: str  # Name of the bone (== Bone.name at the same index) this particle drives
        damping: float
        movementForceFactor: float
        invertedSimWeight: float  # 0 = fully simulated, 1 = pinned to its bone
        isZeroWeight: int
        isOneWeight: int
        colliderBitflag: int
        radius: float  # None if this particle has no radius entry

        def __str__(self):
            # type: () -> str
            return str({slot: getattr(self, slot, None) for slot in self.__slots__})

    class Edge:
        __slots__ = ('node1', 'node2', 'restLength', 'minLength', 'maxLength', 'springStrength', 'movement')

        node1: int  # Index into particles/bones
        node2: int
        restLength: float
        minLength: float  # Same as restLength on version 1 files (no minLength field)
        maxLength: float
        springStrength: float
        movement: int  # 0 = only node2 moves, 2 = both move, 4 = only node1 moves, 5 = neither moves

        def __str__(self):
            # type: () -> str
            return str({slot: getattr(self, slot, None) for slot in self.__slots__})

    class Collider:
        __slots__ = ('rotation', 'position', 'parent', 'cls', 'radius', 'height', 'friction')

        rotation: List[float]  # quaternion (x, y, z, w)
        position: List[float]  # (x, y, z)
        parent: str  # Name of the (real skeleton) bone this collider is attached to
        cls: int  # 0 = plane, 1 = sphere, 2 = capsule, 3 = dual-capsule
        radius: float
        height: float
        friction: float

        def __str__(self):
            # type: () -> str
            return str({slot: getattr(self, slot, None) for slot in self.__slots__})

    class Triangle:
        __slots__ = ('index1', 'index2', 'index3', 'colliderBitflag')

        index1: int
        index2: int
        index3: int
        colliderBitflag: int

        def __str__(self):
            # type: () -> str
            return str({slot: getattr(self, slot, None) for slot in self.__slots__})

    version: int  # 1 = 1.0 format, 2 = extended format (adds edge minLength), 3 = 64-bit format (7.2.1+)
    gravity: List[float]
    boundingSphereRadiusSq: float
    bones: List['Cloth.Bone']
    particles: List['Cloth.Particle']
    edges: List['Cloth.Edge']
    colliders: List['Cloth.Collider']
    triangles: List['Cloth.Triangle']

    def __init__(self):
        self.version = 0
        self.gravity = [0.0, 0.0, 0.0]
        self.boundingSphereRadiusSq = 0.0
        self.bones = []
        self.particles = []
        self.edges = []
        self.colliders = []
        self.triangles = []