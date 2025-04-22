import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.spatial.distance import euclidean


def detect_stars(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img_blur, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stars = [cv2.moments(cnt) for cnt in contours]
    centers = [
        (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        for m in stars if m['m00'] != 0
    ]
    return centers


def triangle_descriptor(p1, p2, p3):
    # Sort points for consistency
    pts = sorted([p1, p2, p3])
    a = euclidean(pts[0], pts[1])
    b = euclidean(pts[1], pts[2])
    c = euclidean(pts[2], pts[0])
    sides = sorted([a, b, c])
    return [sides[0] / sides[2], sides[1] / sides[2]]


def build_triangle_index(stars):
    triangle_data = []
    for tri in itertools.combinations(range(len(stars)), 3):
        i, j, k = tri
        desc = triangle_descriptor(stars[i], stars[j], stars[k])
        triangle_data.append((desc, tri))
    return triangle_data


def match_triangles(index1, index2, threshold=0.05):
    matches = []
    for desc1, tri1 in index1:
        for desc2, tri2 in index2:
            if np.linalg.norm(np.array(desc1) - np.array(desc2)) < threshold:
                matches.append((tri1, tri2))
    return matches


def vote_star_matches(triangle_matches):
    votes = {}
    for t1, t2 in triangle_matches:
        for i, j in zip(t1, t2):
            votes[(i, j)] = votes.get((i, j), 0) + 1

    final_matches = []
    used1, used2 = set(), set()
    for (i, j), count in sorted(votes.items(), key=lambda x: -x[1]):
        if i not in used1 and j not in used2:
            final_matches.append((i, j))
            used1.add(i)
            used2.add(j)
    return final_matches


def visualize_matches(img1_path, stars1, img2_path, stars2, matches):
    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

    h = max(img1.shape[0], img2.shape[0])
    combined = np.zeros((h, img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    combined[:img1.shape[0], :img1.shape[1]] = img1
    combined[:img2.shape[0], img1.shape[1]:] = img2

    for i, (idx1, idx2) in enumerate(matches):
        pt1 = stars1[idx1]
        pt2 = (stars2[idx2][0] + img1.shape[1], stars2[idx2][1])
        cv2.line(combined, pt1, pt2, (0, 255, 255), 5)
        cv2.putText(combined, str(i), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined, str(i), pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.figure(figsize=(16, 8))
    plt.imshow(combined)
    plt.axis('off')
    plt.title("Triangle-Based Matched Stars")
    plt.show()

if __name__ == '__main__':
    image1_path = r'C:\Users\abode\Desktop\computer science\third year\Space Engineering\EX1\1.png'  
    image2_path = r'C:\Users\abode\Desktop\computer science\third year\Space Engineering\EX1\2.png'

    stars1 = detect_stars(image1_path)
    stars2 = detect_stars(image2_path)

    tri_index1 = build_triangle_index(stars1)
    tri_index2 = build_triangle_index(stars2)

    tri_matches = match_triangles(tri_index1, tri_index2)
    matches = vote_star_matches(tri_matches)

    print(f"Found {len(matches)} matched stars:")
    for i1, i2 in matches:
        print(f"Image1: {stars1[i1]} <-> Image2: {stars2[i2]}")

    visualize_matches(image1_path, stars1, image2_path, stars2, matches)
