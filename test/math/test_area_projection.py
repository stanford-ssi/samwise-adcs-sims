import pytest
import numpy as np
from simwise.data_structures.parameters import Parameters
from simwise.math.area_projection import (
    define_satellite_vertices_simple,
    define_satellite_vertices,
    project_prism,
    calculate_projected_area,
    create_rotation_matrix,
    plot_rotation_and_projection,
    plot_rotation_and_projection_cube
)

# Test cases for both simple and complex shapes
@pytest.mark.parametrize("test_case", [
    pytest.param({
        "name": "Unrotated shape",
        "psi": 0, "theta": 0, "phi": 0,
    }, id="unrotated"),
    pytest.param({
        "name": "45-degree rotation around z-axis",
        "psi": np.pi/4, "theta": 0, "phi": 0,
    }, id="z_rotation_45"),
    pytest.param({
        "name": "90-degree rotation around y-axis",
        "psi": 0, "theta": np.pi/2, "phi": 0,
    }, id="y_rotation_90"),
    pytest.param({
        "name": "45-degree rotation around x-axis",
        "psi": 0, "theta": 0, "phi": np.pi/4,
    }, id="x_rotation_45"),
    pytest.param({
        "name": "Complex rotation",
        "psi": np.pi/6, "theta": np.pi/4, "phi": np.pi/3,
    }, id="complex_rotation"),
    pytest.param({
        "name": "90-90-90 rotation",
        "psi": np.pi/2, "theta": np.pi/2, "phi": np.pi/2,
    }, id="90_90_90_rotation"),
])

class TestAreaProjection:
    @pytest.fixture
    def params(self):
        return Parameters()

    def test_simple_cube_area(self, params, test_case):
        """Test that a simple cube maintains consistent projected area under rotation."""
        original_vertices = define_satellite_vertices_simple(params)
        rotated_vertices, projected_vertices = project_prism(
            original_vertices, 
            test_case["psi"], 
            test_case["theta"], 
            test_case["phi"]
        )
        
        calculated_area = calculate_projected_area(projected_vertices)
        expected_area = 1.0
        
        # Plot the rotation and projection
        plot_rotation_and_projection_cube(
            original_vertices, 
            rotated_vertices, 
            projected_vertices,
            f"Simple Cube - {test_case['name']}"
        )
        
        print(f"\nTest case: {test_case['name']}")
        print(f"Expected area: {expected_area:.6f}")
        print(f"Calculated area: {calculated_area:.6f}")
        print(f"Difference: {abs(calculated_area - expected_area):.6f}")
        
        assert np.isclose(calculated_area, expected_area, rtol=1e-5), \
            f"Failed for {test_case['name']}: expected {expected_area}, got {calculated_area}"

    def test_complex_satellite_area(self, params, test_case):
        """Test that the complex satellite shape maintains reasonable projected areas."""
        original_vertices = define_satellite_vertices(params)
        _, reference_projection = project_prism(original_vertices, 0, 0, 0)
        reference_area = calculate_projected_area(reference_projection)

        rotated_vertices, projected_vertices = project_prism(
            original_vertices,
            test_case["psi"],
            test_case["theta"],
            test_case["phi"]
        )
        
        calculated_area = calculate_projected_area(projected_vertices)
        
        # Plot the rotation and projection
        plot_rotation_and_projection(
            original_vertices, 
            rotated_vertices, 
            projected_vertices,
            f"Complex Satellite - {test_case['name']}"
        )
        
        print(f"\nTest case: {test_case['name']}")
        print(f"Reference area: {reference_area:.6f}")
        print(f"Calculated area: {calculated_area:.6f}")
        print(f"Area ratio: {calculated_area/reference_area:.2%}")
        
        assert calculated_area <= reference_area * 1.01, \
            f"Failed for {test_case['name']}: projected area ({calculated_area}) larger than reference ({reference_area})"
        assert calculated_area > 0, \
            f"Failed for {test_case['name']}: projected area is zero or negative ({calculated_area})"

class TestRotationMatrix:
    @pytest.mark.parametrize("test_case", [
        pytest.param({
            "name": "Identity matrix",
            "psi": 0, "theta": 0, "phi": 0,
            "expected": np.eye(3)
        }, id="identity"),
        pytest.param({
            "name": "90-degree z-rotation",
            "psi": np.pi/2, "theta": 0, "phi": 0,
            "expected": np.array([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ])
        }, id="z_90"),
        pytest.param({
            "name": "90-degree y-rotation",
            "psi": 0, "theta": np.pi/2, "phi": 0,
            "expected": np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]
            ])
        }, id="y_90"),
        pytest.param({
            "name": "90-degree x-rotation",
            "psi": 0, "theta": 0, "phi": np.pi/2,
            "expected": np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])
        }, id="x_90"),
    ])
    def test_rotation_matrix(self, test_case):
        """Test that rotation matrices are correctly generated."""
        result = create_rotation_matrix(
            test_case["psi"],
            test_case["theta"],
            test_case["phi"]
        )
        assert np.allclose(result, test_case["expected"], rtol=1e-10), \
            f"Failed for {test_case['name']}"

class TestVertexDefinitions:
    @pytest.fixture
    def params(self):
        return Parameters()

    def test_simple_cube_properties(self, params):
        """Test that the simple cube vertices form a proper unit cube."""
        vertices = define_satellite_vertices_simple(params)
        
        # Test number of vertices
        assert len(vertices) == 8, "Cube should have 8 vertices"
        
        # Test that vertices form a cube centered at origin
        assert np.allclose(np.mean(vertices, axis=0), [0, 0, 0]), \
            "Cube should be centered at origin"
        
        # Test that all vertices are at distance 0.5√3 from origin (unit cube)
        distances = np.linalg.norm(vertices, axis=1)
        expected_distance = 0.5 * np.sqrt(3)
        assert np.allclose(distances, expected_distance), \
            "All vertices should be equidistant from origin"

    def test_satellite_vertices_properties(self, params):
        """Test that the satellite vertices form a valid shape."""
        vertices = define_satellite_vertices(params)
        
        # Test basic properties
        assert len(vertices) == 16, "Satellite should have 16 vertices"
        
        # Test symmetry of the main cube part (first 8 vertices)
        cube_vertices = vertices[:8]
        assert np.allclose(np.mean(cube_vertices, axis=0), [0, 0, 0]), \
            "Main cube should be centered at origin"
        
        # Test solar panel extension
        right_panel_extent = np.max(vertices[:, 0])
        left_panel_extent = np.min(vertices[:, 0])
        assert np.isclose(right_panel_extent, -(left_panel_extent)), \
            "Solar panels should extend symmetrically"
        assert right_panel_extent > CUBESAT_WIDTH/2, \
            "Solar panels should extend beyond cube width"

class TestProjection:
    @pytest.fixture
    def unit_square(self):
        """Returns vertices of a unit square in the YZ plane."""
        return np.array([
            [0, -0.5, -0.5],
            [0, -0.5, 0.5],
            [0, 0.5, -0.5],
            [0, 0.5, 0.5]
        ])

    def test_projection_preserves_area(self, unit_square):
        """Test that projecting an already-flat shape preserves its area."""
        rotated_vertices, projected_vertices = project_prism(unit_square, 0, 0, 0)
        area = calculate_projected_area(projected_vertices)
        
        # Plot the rotation and projection
        plot_rotation_and_projection_cube(
            unit_square, 
            rotated_vertices, 
            projected_vertices,
            "Unit Square Projection Test"
        )
        
        print("\nUnit Square Projection:")
        print(f"Expected area: 1.000000")
        print(f"Calculated area: {area:.6f}")
        print(f"Difference: {abs(area - 1.0):.6f}")
        
        assert np.isclose(area, 1.0, rtol=1e-10), \
            "Projection of unit square should have area 1.0"

    def test_projected_area_bounds(self, params):
        """Test that projected areas are within theoretical bounds."""
        vertices = define_satellite_vertices(params)
        _, ref_projected = project_prism(vertices, 0, 0, 0)
        max_area = calculate_projected_area(ref_projected)
        
        print(f"\nReference maximum area: {max_area:.6f}")
        
        np.random.seed(42)  # For reproducibility
        for i in range(10):
            angles = np.random.uniform(0, 2*np.pi, 3)
            rotated_vertices, projected_vertices = project_prism(vertices, *angles)
            area = calculate_projected_area(projected_vertices)
            
            # Plot every other rotation to avoid too many plots
            if i % 2 == 0:
                plot_rotation_and_projection(
                    vertices,
                    rotated_vertices,
                    projected_vertices,
                    f"Random Rotation {i+1} - Angles: [{angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}]"
                )
            
            print(f"\nRandom rotation {i+1}:")
            print(f"Angles (rad): [{angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}]")
            print(f"Projected area: {area:.6f}")
            print(f"Area ratio: {area/max_area:.2%}")
            
            assert area > 0, "Projected area should be positive"
            assert area <= max_area * 1.01, \
                "Projected area should not exceed front-facing area"

