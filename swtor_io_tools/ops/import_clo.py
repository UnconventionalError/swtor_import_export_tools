# <pep8 compliant>

"""
This script imports Star Wars: The Old Republic cloth physics files into Blender
and builds a real, positioned skeleton (armature) from them.

Usage:
Run this script from "File->Import" menu and then load the desired CLO file.

If the active object when importing is an ARMATURE (e.g. one already produced by
this add-on's .gr2 importer for the matching character skeleton), the cloth bones
are added into that same armature and automatically parented to the real skeleton
bones they reference by name (Pelvis, Chest2, Head, ...). Otherwise, a new,
standalone armature is created containing just the cloth bones (any bone whose
parent is a real skeleton bone -- not present in the .clo file -- is left
unparented, with the intended parent name stored on it as the custom property
"clo_missing_parent").

https://github.com/SWTOR-Slicers/WikiPedia/wiki/CLO-File-Structure

Binary layout ported from Jedipedia's clo_binary-read.js (the actual in-browser
.clo parser), which documents the format far more completely than this add-on's
previous reader did: a fixed 0x10-byte header, then a version-dependent offset
table (32-bit offsets for versions 1-2, 64-bit for version 3 / the 64-bit client),
a fixed-size (0x20 bytes) string table, and five parallel sections: bones,
particles, per-particle radius data, edges, colliders and triangles.
"""

import os
from array import array
from math import pi as PI
from typing import Optional, Set

from bpy import app
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, StringProperty
from bpy.types import Context, Object, Operator, OperatorFileListElement
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Quaternion, Vector

from ..types.clo import Cloth
from ..utils.binary import ArrayBuffer, DataView
from ..utils.string import readString


class ImportCLO(Operator, ImportHelper):
    """Import SWTOR CLO file format (.clo)"""
    bl_idname = "import_cloth.clo"
    bl_label = "Import SWTOR (.clo)"
    bl_description = (
        "Import SWTOR cloth physics bones and build a positioned skeleton from them.\n\n"
        "Select the character's .gr2 skeleton armature first (as the active object) to "
        "merge the cloth bones straight into it, parented to the matching real bones "
        "(Pelvis, Chest2, Head, ...). With nothing suitable selected, a standalone "
        "armature is created instead"
    )
    bl_options = {'UNDO'}

    if app.version < (2, 82, 0):
        directory = StringProperty(subtype='DIR_PATH')
    else:
        directory: StringProperty(subtype='DIR_PATH')

    filename_ext = ".clo"

    files: CollectionProperty(
        name="File Path",
        description="File path used for importing the CLO file",
        type=OperatorFileListElement,
    )
    filter_glob: StringProperty(
        default="*.clo",
        options={'HIDDEN'},
    )

    mode: EnumProperty(
        name="Mode",
        items=(
            ('RIG', "Build Rig", "Build a positioned skeleton from the cloth bones"),
            ('PHYSICS', "Build Cloth Physics", "Set up Cloth Physics on the matching mesh using the .clo data"),
        ),
        default='RIG',
    )
    skip_unweighted_bones: BoolProperty(
        name="Skip Unweighted Bones",
        description="Leave out bones with no weight on the matching mesh (reparenting their children up a level)",
        default=True,
    )
    use_spline_ik: BoolProperty(
        name="Add Spline IK",
        description="Not yet implemented -- will drive each unbranched chain from a generated curve",
        default=False,
    )
    use_bendy_bones: BoolProperty(
        name="Use Bendy Bone segments",
        description="Not yet implemented -- smooths chain segments, usable with or without Spline IK",
        default=False,
    )
    add_master_bone: BoolProperty(
        name="Add Master Bone",
        description=(
            "Add one extra convenience bone above the real-bone placeholders, for a single "
            "grab point or constraint. The placeholders stay independently constrainable "
            "either way -- this only adds a parent above them, it doesn't replace them"
        ),
        default=False,
    )

    def draw(self, context):
        # type: (Context) -> None
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Import Mode")
        box.prop(self, "mode", expand=True)

        if self.mode == 'RIG':
            box = layout.box()
            box.label(text="Rig Options")
            box.prop(self, "skip_unweighted_bones")
            box.prop(self, "use_spline_ik")
            box.prop(self, "use_bendy_bones")
            target = context.active_object
            if not (target and target.type == 'ARMATURE'):
                box.prop(self, "add_master_bone")
        else:
            box = layout.box()
            box.label(text="Cloth Physics Options")
            box.label(text="Not yet implemented", icon='INFO')

    def execute(self, context):
        # type: (Context) -> Set[str]
        paths = [os.path.join(self.directory, file.name) for file in self.files]

        if not paths:
            paths.append(self.filepath)

        for path in paths:
            if not load(self, context, path):
                return {'CANCELLED'}

        return {'FINISHED'}


# ---------------------------------------------------------------------------------
# Binary reading

def _read_fixed_string(dv, pos, size=0x20):
    # type: (DataView, int, int) -> str
    """Reads a null-terminated string out of a fixed-size (size-byte) slot."""
    chars = []
    for i in range(size):
        b = dv.getUint8(pos + i)
        if b == 0:
            break
        chars.append(chr(b))
    return ''.join(chars)


def _fix_crlf_if_needed(raw):
    # type: (bytes) -> bytes
    """
    Some .clo files got every 0x0A byte widened to 0x0D 0x0A somewhere in their asset
    pipeline, making the file 26 bytes longer than its own header says it should be.
    When that mismatch is detected, undo it before parsing (ported from
    clo_binary-read.js, which hits this on real files such as
    chest_capetightskin02_bfa_light_jw_mtx09_shoulder.clo).
    """
    if len(raw) < 16:
        return raw
    payload_offset = int.from_bytes(raw[0x8:0xC], 'little')
    payload_length = int.from_bytes(raw[0xC:0x10], 'little')
    if len(raw) - payload_offset - payload_length == 26:
        return raw.replace(b'\x0d\x0a', b'\x0a')
    return raw


def read(operator, filepath):
    # type: (Operator, str) -> Optional[Cloth]
    with open(filepath, 'rb') as file:
        raw = file.read()

    raw = _fix_crlf_if_needed(raw)

    buffer = ArrayBuffer(len(raw))
    buffer[:] = array('B', raw)
    dv = DataView(buffer)

    # NOTE: Header.
    magic = dv.getUint32(0x0, True)
    if magic != 0x42434C4F:  # b'OLCB'
        operator.report({'ERROR'}, f"\'{filepath}\' is not a valid SWTOR .clo file (bad magic).")
        return None

    cloth = Cloth()
    cloth.version = dv.getUint32(0x4, True)
    if not (1 <= cloth.version <= 3):
        operator.report({'ERROR'}, f"\'{filepath}\': unsupported .clo version {cloth.version} (expected 1-3).")
        return None

    payload_offset = dv.getUint32(0x8, True)
    if payload_offset != 0x10:
        operator.report({'ERROR'}, f"\'{filepath}\': unexpected payload offset {payload_offset} (expected 16).")
        return None
    payload_length = dv.getUint32(0xC, True)
    if dv.byteLength - payload_offset != payload_length:
        operator.report({'WARNING'}, f"\'{filepath}\': payload length mismatch; file may be truncated.")

    cloth.gravity = [dv.getFloat32(0x10, True), dv.getFloat32(0x14, True), dv.getFloat32(0x18, True)]
    cloth.boundingSphereRadiusSq = dv.getFloat32(0x24, True)

    # NOTE: Offset table. Versions 1-2 store 32-bit count/offset pairs interleaved;
    # version 3 (64-bit client, 7.2.1+) stores all counts (32-bit) first, then all
    # offsets as 64-bit (we only need the lower 32 bits -- files are always far
    # smaller than 4 GiB).
    offsets = {}
    if cloth.version < 3:
        offsets['stringsCount'] = dv.getUint32(0x38, True)
        offsets['stringsOffset'] = dv.getUint32(0x3C, True)
        offsets['bonesCount'] = dv.getUint32(0x40, True)
        offsets['bonesOffset'] = dv.getUint32(0x44, True)
        offsets['particlesCount'] = dv.getUint32(0x48, True)
        offsets['particlesOffset'] = dv.getUint32(0x4C, True)
        offsets['particleDataCount'] = dv.getUint32(0x54, True)
        offsets['particleDataOffset'] = dv.getUint32(0x58, True)
        offsets['edgesCount'] = dv.getUint32(0x5C, True)
        offsets['edgesOffset'] = dv.getUint32(0x60, True)
        offsets['collidersCount'] = dv.getUint32(0x64, True)
        offsets['collidersOffset'] = dv.getUint32(0x68, True)
        offsets['trianglesCount'] = dv.getUint32(0x74, True)
        offsets['trianglesOffset'] = dv.getUint32(0x78, True)
    else:
        offsets['stringsCount'] = dv.getUint32(0x38, True)
        offsets['bonesCount'] = dv.getUint32(0x3C, True)
        offsets['particlesCount'] = dv.getUint32(0x40, True)
        offsets['particleDataCount'] = dv.getUint32(0x44, True)
        offsets['edgesCount'] = dv.getUint32(0x48, True)
        offsets['collidersCount'] = dv.getUint32(0x4C, True)
        offsets['trianglesCount'] = dv.getUint32(0x54, True)
        offsets['stringsOffset'] = dv.getUint32(0x58, True)
        offsets['bonesOffset'] = dv.getUint32(0x60, True)
        offsets['particlesOffset'] = dv.getUint32(0x68, True)
        offsets['particleDataOffset'] = dv.getUint32(0x78, True)
        offsets['edgesOffset'] = dv.getUint32(0x80, True)
        offsets['collidersOffset'] = dv.getUint32(0x88, True)
        offsets['trianglesOffset'] = dv.getUint32(0x98, True)

    # NOTE: String table (fixed 0x20-byte slots). Bone/particle/collider names and
    # parent-bone references are all indices into this table.
    strings = []
    for i in range(offsets['stringsCount']):
        pos = 0x10 + offsets['stringsOffset'] + i * 0x20
        strings.append(_read_fixed_string(dv, pos))

    def string_at(index):
        return strings[index] if 0 <= index < len(strings) else ''

    # NOTE: Bones. 0x60-byte stride:
    #   float32[4] boneToParentRot (x, y, z, w)
    #   float32[4] boneToParentTrans (x, y, z, w-is-a-copy-of-x)
    #   float32[4] rootToBoneRot (x, y, z, w)      <- inverse-bind rotation
    #   float32[4] rootToBoneTrans (x, y, z, w-is-a-copy-of-x)  <- inverse-bind translation
    #   float32[4] restEdgeDirection (x, y, z, padding)
    #   int32 startParticle, int32 endParticle
    #   int32 nameIndex, int32 parentNameIndex
    for i in range(offsets['bonesCount']):
        pos = 0x10 + offsets['bonesOffset'] + i * 0x60
        bone = Cloth.Bone()
        bone.index = i
        bone.boneToParentRot = [dv.getFloat32(pos + o, True) for o in (0x0, 0x4, 0x8, 0xC)]
        bone.boneToParentTrans = [dv.getFloat32(pos + o, True) for o in (0x10, 0x14, 0x18)]
        bone.rootToBoneRot = [dv.getFloat32(pos + o, True) for o in (0x20, 0x24, 0x28, 0x2C)]
        bone.rootToBoneTrans = [dv.getFloat32(pos + o, True) for o in (0x30, 0x34, 0x38)]
        bone.restEdgeDirection = [dv.getFloat32(pos + o, True) for o in (0x40, 0x44, 0x48)]
        bone.startParticle = dv.getInt32(pos + 0x50, True)
        bone.endParticle = dv.getInt32(pos + 0x54, True)
        bone.name = string_at(dv.getInt32(pos + 0x58, True))
        bone.parent = string_at(dv.getInt32(pos + 0x5C, True))
        cloth.bones.append(bone)

    # NOTE: Particles. 0x24-byte stride.
    for i in range(offsets['particlesCount']):
        pos = 0x10 + offsets['particlesOffset'] + i * 0x24
        particle = Cloth.Particle()
        particle.index = i
        particle.damping = dv.getFloat32(pos, True)
        particle.movementForceFactor = dv.getFloat32(pos + 0x4, True)
        particle.drivenBone = string_at(dv.getInt32(pos + 0x8, True))
        particle.invertedSimWeight = dv.getFloat32(pos + 0xC, True)
        particle.isZeroWeight = dv.getUint8(pos + 0x10)
        particle.isOneWeight = dv.getUint8(pos + 0x11)
        particle.colliderBitflag = dv.getUint32(pos + 0x14, True)
        particle.radius = None
        cloth.particles.append(particle)

    # NOTE: Per-particle radius data is a sparse side-table; stride differs on v3.
    particle_data_stride = 0x28 if cloth.version == 3 else 0x1C
    for i in range(offsets['particleDataCount']):
        pos = 0x10 + offsets['particleDataOffset'] + i * particle_data_stride
        particle_index = dv.getUint32(pos, True)
        radius = dv.getFloat32(pos + 0x8, True)
        if 0 <= particle_index < len(cloth.particles):
            cloth.particles[particle_index].radius = radius

    # NOTE: Edges. Version 1 has no minLength field (0x18-byte stride); 2+ does (0x1C).
    edge_stride = 0x1C if cloth.version > 1 else 0x18
    for i in range(offsets['edgesCount']):
        pos = 0x10 + offsets['edgesOffset'] + i * edge_stride
        edge = Cloth.Edge()
        edge.node1 = dv.getUint32(pos, True)
        edge.node2 = dv.getUint32(pos + 0x4, True)
        edge.restLength = dv.getFloat32(pos + 0x8, True)
        edge.maxLength = dv.getFloat32(pos + 0xC, True)
        edge.springStrength = dv.getFloat32(pos + 0x10, True)
        edge.movement = dv.getUint32(pos + 0x14, True)
        edge.minLength = dv.getFloat32(pos + 0x18, True) if cloth.version > 1 else edge.restLength
        cloth.edges.append(edge)

    # NOTE: Colliders. 0x40-byte stride.
    for i in range(offsets['collidersCount']):
        pos = 0x10 + offsets['collidersOffset'] + i * 0x40
        collider = Cloth.Collider()
        collider.rotation = [dv.getFloat32(pos + o, True) for o in (0x0, 0x4, 0x8, 0xC)]
        collider.position = [dv.getFloat32(pos + o, True) for o in (0x10, 0x14, 0x18)]
        collider.parent = string_at(dv.getUint32(pos + 0x20, True))
        collider.cls = dv.getUint32(pos + 0x28, True)
        collider.radius = dv.getFloat32(pos + 0x2C, True)
        collider.height = dv.getFloat32(pos + 0x30, True)
        collider.friction = dv.getFloat32(pos + 0x34, True)
        cloth.colliders.append(collider)

    # NOTE: Triangles. 0x10-byte stride. Not used for collision against the real mesh --
    # these approximate it for performance, and only matter for future collider display.
    for i in range(offsets['trianglesCount']):
        pos = 0x10 + offsets['trianglesOffset'] + i * 0x10
        triangle = Cloth.Triangle()
        triangle.index1 = dv.getUint32(pos, True)
        triangle.index2 = dv.getUint32(pos + 0x4, True)
        triangle.index3 = dv.getUint32(pos + 0x8, True)
        triangle.colliderBitflag = dv.getUint32(pos + 0xC, True)
        cloth.triangles.append(triangle)

    return cloth


# ---------------------------------------------------------------------------------
# Armature building

def _bone_rest_position(bone):
    # type: (Cloth.Bone) -> Vector
    """
    The bone's rest position in the coordinate space of the matching .gr2 model.

    rootToBoneRot/rootToBoneTrans is the bind-pose (root/object space -> bone space)
    transform, so its inverse maps the bone's own origin back into object space:
        pos = conj(rootToBoneRot) . rootToBoneTrans
    (verified against the file: for the common case of an identity rootToBoneRot this
    reduces to -rootToBoneTrans, and it reproduces smooth, mirrored bone chains on real
    files -- see the worked example in chat).

    A .clo file stores this in "cloth space", which is every axis of the sibling .gr2
    model's own (Y-up) space negated (Jedipedia's clo-render.js: X-flipped / Y-down /
    Z-flipped). We undo that here so the result lines up with a .gr2-imported skeleton's
    raw (pre axis-conversion) bone positions.
    """
    x, y, z, w = bone.rootToBoneRot
    conjugate = Quaternion((w, -x, -y, -z))  # mathutils quaternions are (w, x, y, z)
    cloth_space = conjugate @ Vector(bone.rootToBoneTrans)
    return Vector((-cloth_space.x, -cloth_space.y, -cloth_space.z))


def _parent_rest_position(bone):
    # type: (Cloth.Bone) -> Vector
    """
    Reconstructs the rest-pose position of BONE's real-skeleton parent (e.g. "Pelvis"),
    even though that bone isn't stored in this .clo file at all.

    boneToParentRot/boneToParentTrans is this bone's own bind-pose placement relative to
    its parent (same "root to local" shape as rootToBoneRot/Trans, one level up), so the
    same inversion recovers the parent's offset in THIS BONE's local space. Rotating that
    offset by this bone's own bone-to-root rotation (the inverse of rootToBoneRot) lands
    it in the same object space `_bone_rest_position` uses, so it can just be added to the
    bone's own rest position.

    Verified against real files: every cloth bone sharing the same named parent
    reconstructs the exact same position for it (zero variance across 5-24 samples).
    """
    bx, by, bz, bw = bone.boneToParentRot
    parent_conjugate = Quaternion((bw, -bx, -by, -bz))
    offset_in_bone_space = parent_conjugate @ Vector(bone.boneToParentTrans)

    rx, ry, rz, rw = bone.rootToBoneRot
    bone_to_root_rotation = Quaternion((rw, -rx, -ry, -rz))
    offset_in_root_space = bone_to_root_rotation @ offset_in_bone_space

    # Same cloth-space -> .gr2-space axis flip as _bone_rest_position, applied to the
    # offset before combining (the flip is linear, so flip(a) + flip(b) == flip(a + b)).
    offset_flipped = Vector((-offset_in_root_space.x, -offset_in_root_space.y, -offset_in_root_space.z))
    return _bone_rest_position(bone) + offset_flipped


AXIS_CONVERSION_ROTATION = Matrix.Rotation(PI * 0.5, 4, 'X')
TAIL_LENGTH = 0.01
PIN_THRESHOLD = 0.999  # invertedSimWeight at/above this counts as a true, rigid anchor.


def _find_pins(cloth):
    # type: (Cloth) -> List[Cloth.Bone]
    """
    True anchor points: particles the sim never moves at all. NOT the same as "every bone
    that references a different real bone" -- most bones do that just to borrow a nearby
    real bone's local coordinate frame while still being freely simulated (see the cape's
    knee/hip references in chat). Only invertedSimWeight ~= 1 means genuinely rigid.
    """
    return [bone for bone in cloth.bones if cloth.particles[bone.index].invertedSimWeight >= PIN_THRESHOLD]


def _spanning_tree_from_pins(cloth, pins, unweighted=frozenset()):
    # type: (Cloth, List[Cloth.Bone], Set[int]) -> dict
    """
    Multi-source Dijkstra over the edges[] graph, seeded from every pin at once, so each
    non-pin bone is claimed by whichever pin is nearest to it by cumulative REST LENGTH,
    not just fewest hops. Returns {bone_index: parent_index or None}; pins map to None
    (they're roots). Bones unreachable from any pin (should be rare -- none seen across 6
    real files so far) are left out of the dict entirely and fall back to flat real-bone
    parenting by the caller.

    Plain hop-count BFS isn't enough: real files have dense cross-bracing (shear/bend
    springs) alongside the primary structural edges, and a bone can end up EQUALLY few
    hops away via a short genuine structural edge or a much longer incidental brace --
    verified on a real file where a bone had 7 edges, and hop-count BFS picked a 0.021
    cross-brace over the correct 0.011 structural edge one hop earlier, purely because of
    dict/queue ordering. Weighting by restLength makes "nearest" mean physically nearest,
    which is what actually distinguishes a structural edge from a brace in most files.

    bone.startParticle/endParticle turns out to directly name each bone's own designated
    structural partner -- verified against every previously-confirmed-correct chain across
    all 6 files, with zero exceptions -- so those claims are treated as AUTHORITATIVE,
    applied as a direct override after Dijkstra runs rather than just a weight discount.
    A discount alone isn't sufficient: Dijkstra's shortest path is undirected, so even a
    heavily discounted edge can still be reached "from the wrong end" via some unrelated
    cheap detour elsewhere in the graph -- verified on a real file where a pruned side
    branch gave a false shortcut into the middle of an otherwise-correct chain, partially
    reversing it. Only UNAMBIGUOUS claims (exactly one bone names a given target) are
    trusted, and a cycle guard skips any claim that would loop back on itself (not seen on
    any real file, but would otherwise strand a whole chain with no root).

    UNWEIGHTED (pruned) bones are excluded from the graph entirely, not just skipped when
    creating the final bones -- letting a path route THROUGH a bone that won't exist in
    the final rig produces wrong results even for OTHER, unrelated bones: that's exactly
    the false-shortcut case above, where the pruned side branch's own particles happened
    to form a short-cut that was numerically nearest.
    """
    import heapq

    end_claims = _unambiguous_end_claims(cloth, unweighted)
    preferred_edges = {
        (min(claimant, target), max(claimant, target))
        for target, claimant in end_claims.items()
    }

    adjacency = {}
    for edge in cloth.edges:
        if edge.node1 in unweighted or edge.node2 in unweighted:
            continue
        pair = (min(edge.node1, edge.node2), max(edge.node1, edge.node2))
        weight = 1e-6 if pair in preferred_edges else max(edge.restLength, 1e-9)
        adjacency.setdefault(edge.node1, []).append((edge.node2, weight))
        adjacency.setdefault(edge.node2, []).append((edge.node1, weight))

    parent_of = {}
    best_distance = {}
    heap = []
    for pin in pins:
        parent_of[pin.index] = None
        best_distance[pin.index] = 0.0
        heapq.heappush(heap, (0.0, pin.index))

    while heap:
        distance, current = heapq.heappop(heap)
        if distance > best_distance.get(current, float('inf')):
            continue  # stale queue entry, a shorter path was already found
        for neighbour, weight in adjacency.get(current, ()):
            candidate = distance + weight
            if candidate < best_distance.get(neighbour, float('inf')):
                best_distance[neighbour] = candidate
                parent_of[neighbour] = current
                heapq.heappush(heap, (candidate, neighbour))

    def leads_back_to(start, target):
        seen = set()
        current = start
        while current in parent_of and parent_of[current] is not None:
            current = parent_of[current]
            if current == target:
                return True
            if current in seen:
                return False
            seen.add(current)
        return False

    for target, claimant in end_claims.items():
        if not leads_back_to(claimant, target):
            parent_of[target] = claimant

    return parent_of


def _find_sibling_mesh(filepath):
    # type: (str) -> Optional['bpy.types.Object']
    """
    Same convention as Jedipedia's clo_binary.js (gr2Name = filename.replace(/.clo$/, '.gr2'))
    and this add-on's own .gr2 importer's "use file name as object name" option: the mesh
    object sharing this .clo's base filename, if one exists in the scene.
    """
    import bpy
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    mesh_ob = bpy.data.objects.get(base_name)
    return mesh_ob if (mesh_ob is not None and mesh_ob.type == 'MESH') else None


def _find_unweighted_bones(cloth, mesh_ob, pins):
    # type: (Cloth, 'bpy.types.Object', List[Cloth.Bone]) -> Set[int]
    """
    Bone indices with no real influence on MESH_OB: either no vertex group of that name
    exists at all, or the group exists but sums to (essentially) zero weight across every
    vertex. A single pass over the mesh's vertices totals every group at once rather than
    re-scanning per bone.

    Pins are never included here, regardless of their own weight: verified against real
    files that a true pin routinely carries ZERO vertex weight by design (the mesh right
    at a rigid anchor point gets skinned directly to the real bone instead of routing
    through an identity-transform pin copy of it), so judging a pin by its own weight
    prunes the actual structural root while keeping ordinary mid-chain bones -- exactly
    backwards.
    """
    pin_indices = {pin.index for pin in pins}
    totals = [0.0] * len(mesh_ob.vertex_groups)
    for vertex in mesh_ob.data.vertices:
        for group_element in vertex.groups:
            if group_element.group < len(totals):
                totals[group_element.group] += group_element.weight

    unweighted = set()
    for bone in cloth.bones:
        if bone.index in pin_indices:
            continue
        vertex_group = mesh_ob.vertex_groups.get(bone.name)
        if vertex_group is None or totals[vertex_group.index] <= 1e-6:
            unweighted.add(bone.index)
    return unweighted


def _effective_parent(bone_index, tree_parent_of, unweighted):
    # type: (int, dict, Set[int]) -> Optional[int]
    """Walks up the spanning tree past any pruned (unweighted) bones to find the nearest
    surviving ancestor, so skipping a bone doesn't break the chain below it. Pins already
    map to None in tree_parent_of, so they (and anything whose whole ancestry up to a pin
    was pruned) correctly come back as None -- a new effective root."""
    parent_index = tree_parent_of.get(bone_index)
    while parent_index is not None and parent_index in unweighted:
        parent_index = tree_parent_of.get(parent_index)
    return parent_index


def _unambiguous_end_claims(cloth, unweighted=frozenset()):
    # type: (Cloth, Set[int]) -> dict
    """
    {target_bone_index: claimant_bone_index} for every particle with exactly one bone
    naming it as their own endParticle -- OR, when several bones name the same target,
    exactly one of those claimants that ALSO has a genuine edges[] entry to it. Claims
    involving a pruned (unweighted) bone on either side are dropped too -- it won't exist
    in the final rig, so it can't meaningfully be anyone's parent or child.

    Does NOT require a real edge for an otherwise-unambiguous (single-claimant) target.
    endParticle describes the file's own DESIGNED skeletal segments; edges[] describes the
    separate simulation springs -- they don't have to fully overlap, and requiring one was
    a defensive assumption that turned out wrong: verified on a real file (the cape) where
    Bone79's endParticle names Bone81 with no matching edges[] entry at all, and that's
    still the correct relationship (confirmed against the actual expected chain).

    A real edge DOES matter as a tiebreak once a target has multiple claimants, though --
    verified on a real file where a particle was named by 4 different bones' endParticle,
    but only one of those four had any edge to it at all (the other three sat in an
    entirely different part of the mesh, evidently stale or unrelated data); narrowing to
    the edge-having claimant recovers the correct, otherwise-legitimate relationship
    instead of dropping it as unresolvable.
    """
    all_claimants = {}  # target -> list of claimant indices, in encounter order
    for bone in cloth.bones:
        if bone.index in unweighted:
            continue
        if 0 <= bone.endParticle < len(cloth.bones) and bone.endParticle != bone.index:
            if bone.endParticle in unweighted:
                continue
            all_claimants.setdefault(bone.endParticle, []).append(bone.index)

    edge_set = set()
    for edge in cloth.edges:
        if edge.node1 in unweighted or edge.node2 in unweighted:
            continue
        edge_set.add((min(edge.node1, edge.node2), max(edge.node1, edge.node2)))

    claimants = {}
    for target, claimant_list in all_claimants.items():
        if len(claimant_list) == 1:
            claimants[target] = claimant_list[0]
            continue
        with_edge = [c for c in claimant_list if (min(c, target), max(c, target)) in edge_set]
        if len(with_edge) == 1:
            claimants[target] = with_edge[0]
        # else: still genuinely ambiguous (zero or multiple edge-backed claimants) -- drop.
    return claimants


def _seed_pins(cloth, pins, unweighted=frozenset()):
    # type: (Cloth, List[Cloth.Bone], Set[int]) -> List[Cloth.Bone]
    """
    Drops a pin from the seed set only when exactly one OTHER PIN unambiguously names it
    as its own designated continuation (bone.endParticle) -- letting it chain under that
    pin instead of standing as its own independent root.

    Verified against real files this is the precise signal, and nothing broader: it
    exactly reconstructs the shawl's pinned-yoke chains (072->061->050->039 and
    077->066->055->044, each link a genuine pin-claims-pin endParticle reference) while
    leaving the robe's 4 pins alone, since none of them ever reference each other this way
    -- they're independent strands that only happen to share a real parent and sit in the
    same connected component, which an earlier, broader version of this function wrongly
    treated as enough justification to merge them into one seed (confirmed as a real
    regression: it broke newtonparticle002's independent strand on the robe).
    """
    end_claims = _unambiguous_end_claims(cloth, unweighted)
    pin_indices = {pin.index for pin in pins}

    claimed_by_pin = {
        target: claimant
        for target, claimant in end_claims.items()
        if target in pin_indices and claimant in pin_indices
    }

    def leads_to_cycle(start):
        # Guards against pins claiming each other in a loop (not seen on any real file,
        # but would otherwise strand the whole group with no root at all).
        seen = set()
        current = start
        while current in claimed_by_pin:
            current = claimed_by_pin[current]
            if current == start:
                return True
            if current in seen:
                return False
            seen.add(current)
        return False

    de_seeded = {target for target in claimed_by_pin if not leads_to_cycle(target)}
    return [pin for pin in pins if pin.index not in de_seeded]


def build(operator, context, cloth, filepath):
    # type: (Operator, Context, Cloth, str) -> bool
    import bpy

    target_ob = context.active_object
    attaching_to_existing = bool(target_ob and target_ob.type == 'ARMATURE')

    prefs = bpy.context.preferences.addons["swtor_io_tools"].preferences
    scale_factor = prefs.gr2_scale_factor if prefs.gr2_scale_object else 1.0

    if attaching_to_existing:
        armature_ob = target_ob
        # Matches this add-on's .gr2 importer: when it "applies" axis conversion, the
        # rotation gets baked into the edit-bone data itself instead of living on the
        # object, so our raw (un-rotated) positions need the same bake to line up.
        bake_rotation = bool(armature_ob.get("gr2_axis_conversion", False))
        created_new = False
    else:
        armature_data = bpy.data.armatures.new(os.path.splitext(os.path.basename(filepath))[0] + "_skeleton")
        armature_ob = bpy.data.objects.new(armature_data.name, armature_data)
        context.collection.objects.link(armature_ob)
        bake_rotation = False
        created_new = True

    pre_rotation = AXIS_CONVERSION_ROTATION if bake_rotation else Matrix.Identity(4)

    def to_armature_space(v):
        return (pre_rotation @ v.to_4d()).to_3d()

    bpy.ops.object.select_all(action='DESELECT')
    armature_ob.select_set(True)
    context.view_layer.objects.active = armature_ob

    prev_mode = armature_ob.mode
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='EDIT')
    else:
        operator.report({'ERROR'}, "Could not enter Edit Mode on the target armature.")
        return False

    edit_bones = armature_ob.data.edit_bones

    # NOTE on hierarchy: a .clo file is a mass-spring CLOTH MESH, not a bone chain -- see
    # `edges`, where most particles connect to 2-4 neighbours (structural/shear/bend
    # springs), not one. `bone.parent` is the single real skeleton bone a particle
    # borrows its local coordinate frame from -- true for every bone, but only load-
    # bearing (a genuine rigid attachment) for the small subset that are actually PINNED
    # (invertedSimWeight ~= 1). So the rig is rooted at the pins, not at every bone.parent
    # reference, and everything else is positioned by nearest-pin graph distance (a
    # multi-source BFS over `edges`) -- see chat for the cape's collar/chest pins vs. its
    # merely-nearby knee/hip references, which is what this distinction is for.

    pins = _find_pins(cloth)
    pin_indices = {pin.index for pin in pins}

    sibling_mesh = _find_sibling_mesh(filepath)
    if not getattr(operator, "skip_unweighted_bones", True):
        unweighted = set()
    elif sibling_mesh is not None:
        unweighted = _find_unweighted_bones(cloth, sibling_mesh, pins)
        if unweighted:
            operator.report(
                {'INFO'},
                f"Skipping {len(unweighted)} bone(s) with no weight on \'{sibling_mesh.name}\': "
                + ", ".join(sorted(cloth.bones[i].name for i in unweighted)),
            )
    else:
        unweighted = set()
        operator.report(
            {'INFO'},
            f"No mesh named \'{os.path.splitext(os.path.basename(filepath))[0]}\' found in the "
            "scene -- skipping the zero-weight bone check (import the matching .gr2 first to "
            "enable it).",
        )

    # Pruned bones are excluded from the graph itself here, not just skipped afterward --
    # routing another bone's parent assignment THROUGH a bone that won't even exist in the
    # final rig makes no sense, and can produce wrong results even when it's not the one
    # being placed: verified on a real file where a pruned side-branch's own particles
    # formed a short-cut that was numerically "nearest," corrupting an unrelated, otherwise
    # correct chain that merely passed near it.
    tree_parent_of = _spanning_tree_from_pins(cloth, _seed_pins(cloth, pins, unweighted), unweighted)

    effective_parent_of = {
        bone.index: _effective_parent(bone.index, tree_parent_of, unweighted)
        for bone in cloth.bones if bone.index not in unweighted
    }
    effective_children = {}
    for index, parent_index in effective_parent_of.items():
        if parent_index is not None:
            effective_children.setdefault(parent_index, []).append(index)

    def _subtree_size(index, memo={}):
        # type: (int, dict) -> int
        if index in memo:
            return memo[index]
        memo[index] = 1  # guard against any accidental cycle before recursing
        size = 1 + sum(_subtree_size(child) for child in effective_children.get(index, []))
        memo[index] = size
        return size

    primary_child_of = {}  # bone.index -> the one child its tail actually reaches/connects to
    for index, children in effective_children.items():
        primary_child_of[index] = max(children, key=lambda c: (_subtree_size(c), -c))

    skipped_existing = []
    for bone in cloth.bones:
        if bone.index in unweighted:
            continue
        if bone.name in edit_bones:
            skipped_existing.append(bone.name)
            continue
        edit_bones.new(bone.name)

    surviving_pins = [pin for pin in pins if pin.index not in unweighted]

    # One placeholder bone per DISTINCT real-parent name (not per pin) -- e.g. the lekku's
    # two pins both reference "Head", so this creates a single "Head" placeholder, not two;
    # the cape's 8 pins reference 3 real bones, so this creates 3. Each pin parents to the
    # placeholder matching its own real parent, giving independently-posable, independently
    # constrainable anchors instead of one bone standing in for several real bones at once.
    # Position uses the same reconstructed real-bone math validated earlier: exact, not an
    # approximation, whenever every contributing pin agrees (which they always have on
    # every real file tested so far).
    placeholder_of_real_parent = {}
    if not attaching_to_existing:
        representative_pin = {}
        for pin in surviving_pins:
            if pin.parent and pin.parent not in representative_pin:
                representative_pin[pin.parent] = pin
        for real_parent_name, pin in representative_pin.items():
            head = to_armature_space(_parent_rest_position(pin))
            placeholder_eb = edit_bones.new(real_parent_name)
            placeholder_eb.head = head
            placeholder_eb.tail = head + Vector((0, TAIL_LENGTH, 0))
            placeholder_eb["clo_placeholder"] = True
            placeholder_of_real_parent[real_parent_name] = placeholder_eb

    master_eb = None
    if not attaching_to_existing and placeholder_of_real_parent and getattr(operator, "add_master_bone", True):
        # Optional convenience layer ABOVE the accurate placeholders, for a single
        # grab-everything control or a single constraint when per-anchor tracking isn't
        # needed (e.g. quick posing with no matching real skeleton in the scene at all).
        # Placeholders remain independently constrainable either way -- this only adds an
        # extra parent above them, it doesn't replace them.
        centroid = sum(
            (eb.head for eb in placeholder_of_real_parent.values()), Vector((0, 0, 0))
        ) / len(placeholder_of_real_parent)
        master_eb = edit_bones.new(os.path.splitext(os.path.basename(filepath))[0] + "_master")
        master_eb.head = centroid
        master_eb.tail = centroid + Vector((0, TAIL_LENGTH, 0))
        master_eb["clo_master"] = True
        master_eb["clo_real_parents"] = ", ".join(sorted(placeholder_of_real_parent))
        for placeholder_eb in placeholder_of_real_parent.values():
            placeholder_eb.parent = master_eb

    for bone in cloth.bones:
        if bone.index in unweighted:
            continue
        eb = edit_bones.get(bone.name)
        if eb is None:
            continue

        head = to_armature_space(_bone_rest_position(bone))
        eb.head = head

        # Tail: reach exactly to the primary child (full length, not a fixed stub) so the
        # chain reads as one continuous line -- required groundwork for Spline IK/Bendy
        # later, and just looks right for plain FK too. Non-primary children (branch
        # points) still get their own correct head from their own rest position; they're
        # just not "connected" to this bone's tail, same as any normal Blender fork.
        primary = primary_child_of.get(bone.index)
        if primary is not None:
            eb.tail = to_armature_space(_bone_rest_position(cloth.bones[primary]))
        else:
            # Leaf: continue the same direction the chain was already travelling (parent
            # -> self, extrapolated) rather than trusting restEdgeDirection -- verified
            # against real files to point backward toward the anchor, not outward toward
            # the tip (dot product ~= -1.0 with the true outward direction on both tips
            # tested), so it can't be used as an "outward" direction at all.
            parent_index = effective_parent_of.get(bone.index)
            direction = None
            segment_length = TAIL_LENGTH
            if parent_index is not None:
                parent_head = to_armature_space(_bone_rest_position(cloth.bones[parent_index]))
                incoming = head - parent_head
                if incoming.length > 1e-6:
                    direction = incoming.normalized()
                    segment_length = incoming.length
            if direction is None:
                # No parent to extrapolate from either (an isolated pinned particle with
                # nothing attached) -- arbitrary but harmless, there's nothing to get wrong.
                direction = Vector((0.0, 1.0, 0.0))
            eb.tail = head + direction * max(segment_length, TAIL_LENGTH * 0.25)

        # A bone becomes an effective root either because it's a true pin that was chosen
        # to seed the tree, or because every ancestor up to its seed got pruned above (see
        # _effective_parent) -- both get the same root-level (master/real-bone) treatment.
        # Non-seed pins (chained to a nearer pin sharing the same real parent -- see
        # _seed_pins) still get tagged here for reference even though they're not roots;
        # tagging doesn't affect parenting.
        if bone.index in pin_indices:
            eb["clo_real_parent"] = bone.parent

        is_effective_root = effective_parent_of[bone.index] is None
        if is_effective_root:
            if attaching_to_existing:
                parent_eb = edit_bones.get(bone.parent)
                if parent_eb is None and bone.parent:
                    # Merging into a skeleton that's missing this real bone (partial/wrong
                    # armature selected). Rather than leave the pin floating unparented,
                    # stand up a placeholder at its correct position so it's still usable.
                    parent_eb = edit_bones.new(bone.parent)
                    placeholder_head = to_armature_space(_parent_rest_position(bone))
                    parent_eb.head = placeholder_head
                    parent_eb.tail = placeholder_head + Vector((0, TAIL_LENGTH, 0))
                    parent_eb["clo_placeholder"] = True
                    operator.report(
                        {'WARNING'},
                        f"\'{bone.parent}\' isn't in the target armature; added a placeholder "
                        "bone for it at its correct position instead.",
                    )
                if parent_eb is not None:
                    eb.parent = parent_eb
            else:
                parent_eb = placeholder_of_real_parent.get(bone.parent)
                if parent_eb is not None:
                    eb.parent = parent_eb
        else:
            parent_eb = edit_bones.get(cloth.bones[effective_parent_of[bone.index]].name)
            if parent_eb is not None:
                eb.parent = parent_eb

    # Now that every bone has both a parent and a tail reaching its primary child, snap
    # each primary child's head onto that tail exactly (use_connect requires the parent
    # to already be assigned, hence the separate pass after the loop above).
    for parent_index, child_index in primary_child_of.items():
        child_eb = edit_bones.get(cloth.bones[child_index].name)
        if child_eb is not None and child_eb.parent is not None:
            child_eb.use_connect = True

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode=prev_mode if prev_mode != 'EDIT' else 'OBJECT')

    armature_ob.show_in_front = True

    if created_new:
        armature_ob.matrix_local = AXIS_CONVERSION_ROTATION
        armature_ob["gr2_axis_conversion"] = False
        armature_ob.data.display_type = 'STICK'
        armature_ob.scale *= scale_factor
        armature_ob["gr2_scale"] = scale_factor

    if skipped_existing:
        operator.report(
            {'INFO'},
            f"{len(skipped_existing)} bone(s) already existed in the target armature and were "
            "repositioned/reparented rather than duplicated.",
        )

    return True


def load(operator, context, filepath=""):
    # type: (Operator, Context, str) -> bool
    from bpy_extras.wm_utils.progress_report import ProgressReport

    if getattr(operator, "mode", 'RIG') == 'PHYSICS':
        operator.report({'ERROR'}, "Build Cloth Physics isn't implemented yet -- switch Mode to Build Rig.")
        return False

    if getattr(operator, "use_spline_ik", False) or getattr(operator, "use_bendy_bones", False):
        operator.report(
            {'INFO'},
            "Spline IK / Bendy Bone segments aren't implemented yet -- building a plain FK rig.",
        )

    with ProgressReport(context.window_manager) as progress:
        progress.enter_substeps(2, f"Importing \'{filepath}\' ...")

        progress.step("Parsing file ...", 1)
        cloth = read(operator, filepath)

        if cloth:
            progress.step("Building skeleton ...", 2)

            if build(operator, context, cloth, filepath):
                progress.leave_substeps(f"Done, finished importing: \'{filepath}\'")
                return True

        return False