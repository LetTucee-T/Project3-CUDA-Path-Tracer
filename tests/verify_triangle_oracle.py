"""Check the recorded thin-triangle oracle independently with 80-digit arithmetic.

Run with Python 3 from any directory. Uses only the standard library; this is
not a dependency of the renderer or its C++/CUDA regression executables.
"""
from decimal import Decimal, localcontext
import json
from pathlib import Path
import struct


def vector(values):
    # The fixture describes the exact binary32 inputs received by the GPU.
    for value in values:
        assert struct.unpack('<f', struct.pack('<f', value))[0] == value
    return [Decimal.from_float(value) for value in values]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


if __name__ == '__main__':
    data = json.loads((Path(__file__).parent/'data/thin_triangle_regressions.json').read_text())
    hits = 0
    with localcontext() as context:
        context.prec = 80
        for row in data['cases']:
            a, b, c = map(vector, row['vertices'])
            origin, direction = vector(row['origin']), vector(row['direction'])
            # Moller-Trumbore is independent of the production projected edges.
            e1, e2 = sub(b, a), sub(c, a)
            p, offset = cross(direction, e2), sub(origin, a)
            determinant, q = dot(e1, p), cross(offset, e1)
            assert determinant != 0
            u, v, t = dot(offset, p)/determinant, dot(direction, q)/determinant, dot(e2, q)/determinant
            hit = u >= 0 and v >= 0 and u+v <= 1 and t > Decimal('0.00001')
            assert row['expected_hit'] == hit, row['pixel']
            assert row['expected_t'] == (float(t) if hit else -1), row['pixel']
            assert row['expected_barycentrics'] == [float(1-u-v), float(u), float(v)], row['pixel']
            hits += hit
    print(f'80-digit oracle verified: {len(data["cases"])} cases, {hits} hits, {len(data["cases"])-hits} misses.')
