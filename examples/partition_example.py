import numpy as np
import heapq
import argparse
import sys
import polyscope as ps

def container_partition(container_size, n_target, min_ratio, shrink_factor=0.98):
    diag = np.linalg.norm(container_size)
    min_dim = diag * min_ratio
    
    initial_box = (0, container_size[0], 0, container_size[1], 0, container_size[2])
    vol = np.prod(container_size)
    
    heap = [(-vol, initial_box)]
    all_final_boxes = []
    current_count = 1
    
    while current_count < n_target:
        if not heap:
            break
            
        neg_vol, box = heapq.heappop(heap)
        bx0, bx1, by0, by1, bz0, bz1 = box
        dims = [bx1 - bx0, by1 - by0, bz1 - bz0]
        valid_axes = [i for i, d in enumerate(dims) if d >= 2 * min_dim]
        
        if not valid_axes:
            all_final_boxes.append(box)
            continue
            
        axis = valid_axes[np.argmax([dims[i] for i in valid_axes])]
        low = box[axis*2] + min_dim
        high = box[axis*2 + 1] - min_dim
        split = np.random.uniform(low, high)
        
        b1 = list(box); b1[axis*2 + 1] = split
        v1 = (b1[1]-b1[0]) * (b1[3]-b1[2]) * (b1[5]-b1[4])
        
        b2 = list(box); b2[axis*2] = split
        v2 = (b2[1]-b2[0]) * (b2[3]-b2[2]) * (b2[5]-b2[4])
        
        heapq.heappush(heap, (-v1, tuple(b1)))
        heapq.heappush(heap, (-v2, tuple(b2)))
        
        current_count += 1
        
    final_output = all_final_boxes + [b for _, b in heap]
    
    results = []
    for b in final_output:
        center = [(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2]
        lengths = [b[1]-b[0], b[3]-b[2], b[5]-b[4]]
        results.append((center, [l * shrink_factor for l in lengths]))
        
    if len(results) < n_target:
        print(f"Warning: only produced {len(results)} boxes (target was {n_target}), min_ratio too large.")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=float, nargs=3, default=[0.6, 0.4, 0.3])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--min_ratio", type=float, default=0.05)
    parser.add_argument("--shrink", type=float, default=0.98)
    args = parser.parse_args()

    cuboids = container_partition(args.dims, args.n, args.min_ratio, args.shrink)
    if not cuboids:
        sys.exit(1)

    # Initialize Polyscope
    ps.init()
    ps.set_up_dir("z_up")

    # Define the triangulation for a unit cube (12 triangles)
    # Vertices: 0:(-,-,-), 1:(+,-,-), 2:(+,+,-), 3:(-,+,,-), 4:(-,-,+), 5:(+,-,+), 6:(+,+,+), 7:(-,+,+)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [6, 5, 4], [7, 6, 4], # bottom & top
        [1, 5, 6], [1, 6, 2], [0, 4, 7], [0, 7, 3], # sides
        [3, 2, 6], [3, 6, 7], [1, 0, 4], [1, 4, 5]  # sides
    ])

    for i, (center, lengths) in enumerate(cuboids):
        half = np.array(lengths) / 2.0
        c = np.array(center)
        
        # Calculate 8 corner vertices for the current cuboid
        verts = np.array([
            c + [-half[0], -half[1], -half[2]],
            c + [ half[0], -half[1], -half[2]],
            c + [ half[0],  half[1], -half[2]],
            c + [-half[0],  half[1], -half[2]],
            c + [-half[0], -half[1],  half[2]],
            c + [ half[0], -half[1],  half[2]],
            c + [ half[0],  half[1],  half[2]],
            c + [-half[0],  half[1],  half[2]]
        ])
        
        # Register as a surface mesh in Polyscope
        name = f"block_{i}"
        mesh = ps.register_surface_mesh(name, verts, faces)
        
        # Assign a random RGB color
        mesh.set_color(tuple(np.random.rand(3)))

    print(f"Visualization ready. Rendered {len(cuboids)} blocks.")
    
    # Launch the GUI (blocking call)
    ps.show()

if __name__ == "__main__":
    main()