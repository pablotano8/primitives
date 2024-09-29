import pymunk
import numpy as np
import matplotlib.pyplot as plt

class CircularWorld:
    def __init__(self,
                 num_obstacles,
                 max_speed,
                 radius,
                 wall_present=True,
                 obstacle_radius=0.05,
                 wall_size=0.6,
                 wall_thickness=0.2,
                 second_wall=False,
                 fall=True):
        
        self.obstacle_radius = obstacle_radius
        self.radius = radius
        self.wall_present = wall_present
        self.space = pymunk.Space()
        self.max_speed = max_speed
        self.obstacles = []
        self.fallen_off_position = None  # Track position where the agent fell off
        self.wall_size= wall_size
        self.wall_thickness = wall_thickness
        self.second_wall = second_wall
        self.fall = fall

        if wall_present:
            self.add_diameter_wall()
            if self.second_wall:
                self.add_second_wall()
        else:
            self.add_random_obstacles(num_obstacles)

    def add_diameter_wall(self):
        wall_thickness = self.wall_thickness  # Thickness of the wall
        wall_length = self.radius * 2 * self.wall_size # Wall length
        wall_shape = [(-wall_length / 2, -wall_thickness / 2), (wall_length / 2, -wall_thickness / 2),
                      (wall_length / 2, wall_thickness / 2), (-wall_length / 2, wall_thickness / 2)]
        self.add_obstacle(wall_shape)

    def add_second_wall(self):
        wall_thickness = 0.2  # Thickness of the wall
        wall_length = self.radius * 2 * 0.4 # Wall length
        wall_shape = [(-wall_length / 2, -wall_thickness / 2 +0.5), (wall_length / 2, -wall_thickness / 2 +0.5),
                      (wall_length / 2, wall_thickness / 2 +0.5), (-wall_length / 2, wall_thickness / 2 +0.5)]
        self.add_obstacle(wall_shape)

    def add_random_obstacles(self, num_obstacles):
        num_per_row = int(np.sqrt(num_obstacles))
        for i in range(num_per_row):
            for j in range(num_per_row):
                while True:
                    angle = np.random.uniform(0, 2*np.pi)
                    distance = np.random.uniform(0, self.radius)
                    center = (distance * np.cos(angle), distance * np.sin(angle))
                    if np.linalg.norm(center) + self.obstacle_radius <= self.radius:
                        obstacle = self.generate_random_obstacle(center)
                        self.add_obstacle(obstacle)
                        break

    def add_obstacle(self, obstacle):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Poly(body, obstacle)
        self.space.add(body, shape)
        self.obstacles.append((body, shape))

    def generate_random_obstacle(self, center):
        num_vertices = np.random.randint(6, 8)
        obstacle = [self.generate_random_vertex_around(center, self.obstacle_radius) for _ in range(num_vertices)]
        return obstacle

    def generate_random_vertex_around(self, center, radius):
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(0, radius)
        x = center[0] + np.cos(angle) * distance
        y = center[1] + np.sin(angle) * distance
        return (x, y)

    def actual_dynamics(self, position, desired_velocity, dt):
        if self.fallen_off_position is not None:
            # If the agent has fallen off, return the fallen off position
            return self.fallen_off_position, np.array([0.0, 0.0]), True, np.zeros(len(self.obstacles), dtype=bool), np.full(len(self.obstacles), float('inf'))

        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = tuple(position)
        body.velocity = tuple(desired_velocity * dt)

        self.space.add(body)

        collision_detected = False
        collision_normal = None
        for obstacle in self.obstacles:
            query_info = self.space.segment_query_first(body.position, body.position + body.velocity, 0, pymunk.ShapeFilter())
            if query_info:
                collision_detected = True
                collision_normal = query_info.normal
                break

        collision_detected_all_obs = np.zeros(len(self.obstacles), dtype=bool)
        min_distance_all_obs = np.full(len(self.obstacles), float('inf'))
        for i, obstacle in enumerate(self.obstacles):
            query_info = obstacle[1].point_query(pymunk.Vec2d(position[0], position[1]))
            if query_info.distance < min_distance_all_obs[i]:
                min_distance_all_obs[i] = query_info.distance

            query_info = obstacle[1].segment_query(body.position, body.position + body.velocity)
            if query_info.shape is not None:
                collision_detected_all_obs[i] = True

        self.space.remove(body)

        speed = np.linalg.norm(desired_velocity)
        if collision_detected:
            velocity_perpendicular = np.dot(desired_velocity, collision_normal) * collision_normal
            velocity_parallel = desired_velocity - velocity_perpendicular
            actual_velocity = velocity_parallel
        elif speed > self.max_speed:
            actual_velocity = (desired_velocity / speed) * self.max_speed
        else:
            actual_velocity = desired_velocity

        new_position = position + actual_velocity * dt

        if self.fall:
            # Check if the new position is outside the circle's radius
            if np.linalg.norm(new_position) > self.radius:
                self.fallen_off_position = position  # Mark the position where the agent fell off
                new_position = position  # Stick to the edge by not updating the position further
                actual_velocity = np.array([0.0, 0.0])  # Zero velocity on the edge

        return new_position, actual_velocity, collision_detected, collision_detected_all_obs, min_distance_all_obs

    def reset(self):
        self.fallen_off_position = None  # Reset the fallen off position

        while True:
            start = np.random.uniform(-self.radius, self.radius, size=2)
            if np.linalg.norm(start) <= self.radius and not is_inside_obstacle(start, self):
                break

        while True:
            goal = np.random.uniform(-self.radius, self.radius, size=2)
            if np.linalg.norm(goal) <= self.radius and not is_inside_obstacle(goal, self):
                break

        return start, goal

class RandomRectangleWorld:
    def __init__(self,
                 num_obstacles,
                 max_speed,
                 radius,
                 wall_present=True,
                 obstacle_radius=0.05,
                 wall_size=0.6,
                 wall_thickness=0.2,
                 second_wall=False,
                 fall=True,
                 friction=0):
        
        self.obstacle_radius = obstacle_radius
        self.radius = radius
        self.wall_present = wall_present
        self.space = pymunk.Space()
        self.max_speed = max_speed
        self.obstacles = []
        self.fallen_off_position = None  # Track position where the agent fell off
        self.wall_size= wall_size
        self.wall_thickness = wall_thickness
        self.second_wall = second_wall
        self.fall = fall
        self.friction = friction

        if wall_present:
            self.add_diameter_wall()
            if self.second_wall:
                self.add_second_wall()
        else:
            self.add_random_obstacles(num_obstacles)

    def add_diameter_wall(self):
        wall_thickness = self.wall_thickness  # Thickness of the wall
        wall_length = self.radius * 2 * self.wall_size # Wall length
        wall_shape = [(-wall_length / 2, -wall_thickness / 2), (wall_length / 2, -wall_thickness / 2),
                      (wall_length / 2, wall_thickness / 2), (-wall_length / 2, wall_thickness / 2)]
        self.add_obstacle(wall_shape)

    def add_second_wall(self):
        wall_thickness = 0.2  # Thickness of the wall
        wall_length = self.radius * 2 * 0.4 # Wall length
        wall_shape = [(-wall_length / 2, -wall_thickness / 2 +0.5), (wall_length / 2, -wall_thickness / 2 +0.5),
                      (wall_length / 2, wall_thickness / 2 +0.5), (-wall_length / 2, wall_thickness / 2 +0.5)]
        self.add_obstacle(wall_shape)

    def add_random_obstacles(self, num_obstacles):
        num_per_row = int(np.sqrt(num_obstacles))
        for i in range(num_per_row):
            for j in range(num_per_row):
                while True:
                    angle = np.random.uniform(0, 2*np.pi)
                    distance = np.random.uniform(0, self.radius)
                    center = (distance * np.cos(angle), distance * np.sin(angle))
                    if np.linalg.norm(center) + self.obstacle_radius <= self.radius:
                        obstacle = self.generate_random_obstacle(center)
                        self.add_obstacle(obstacle)
                        break

    def add_obstacle(self, obstacle):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Poly(body, obstacle)
        self.space.add(body, shape)
        self.obstacles.append((body, shape))

    def generate_random_obstacle(self, center):
        num_vertices = np.random.randint(6, 8)
        obstacle = [self.generate_random_vertex_around(center, self.obstacle_radius) for _ in range(num_vertices)]
        return obstacle

    def generate_random_vertex_around(self, center, radius):
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(0, radius)
        x = center[0] + np.cos(angle) * distance
        y = center[1] + np.sin(angle) * distance
        return (x, y)

    def actual_dynamics(self, position, desired_velocity, dt):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = tuple(position)
        body.velocity = tuple(desired_velocity * dt)

        self.space.add(body)

        collision_detected = False
        collision_normal = None
        for obstacle in self.obstacles:
            query_info = self.space.segment_query_first(body.position, body.position + body.velocity, 0, pymunk.ShapeFilter())
            if query_info:
                collision_detected = True
                collision_normal = query_info.normal
                break

        collision_detected_all_obs = np.zeros(len(self.obstacles), dtype=bool)
        min_distance_all_obs = np.full(len(self.obstacles), float('inf'))
        for i, obstacle in enumerate(self.obstacles):
            query_info = obstacle[1].point_query(pymunk.Vec2d(position[0], position[1]))
            if query_info.distance < min_distance_all_obs[i]:
                min_distance_all_obs[i] = query_info.distance

            query_info = obstacle[1].segment_query(body.position, body.position + body.velocity)
            if query_info.shape is not None:
                collision_detected_all_obs[i] = True

        self.space.remove(body)

        speed = np.linalg.norm(desired_velocity)
        if collision_detected:
            # Handle collision with obstacles
            velocity_perpendicular = np.dot(desired_velocity, collision_normal) * collision_normal
            velocity_parallel = desired_velocity - velocity_perpendicular
            actual_velocity = velocity_parallel
        elif speed > self.max_speed:
            actual_velocity = (desired_velocity / speed) * self.max_speed
        else:
            actual_velocity = desired_velocity

        new_position = position + actual_velocity * dt

        # Handle collision with circle boundary
        distance_to_center = np.linalg.norm(new_position)
        if distance_to_center > self.radius:
            # Compute the normal vector pointing from the center to the agent's position
            normal = new_position / distance_to_center  # Normalize the vector
            # Project the velocity onto the normal
            velocity_perpendicular = np.dot(actual_velocity, normal) * normal
            # Subtract the perpendicular component to get the velocity parallel to the boundary
            actual_velocity =self.friction *(actual_velocity - velocity_perpendicular)
            # Recompute the new position
            new_position = position + actual_velocity * dt
            # Ensure the new position is inside the circle
            if np.linalg.norm(new_position) > self.radius:
                new_position = new_position / np.linalg.norm(new_position) * self.radius

        return new_position, actual_velocity, collision_detected, collision_detected_all_obs, min_distance_all_obs


    def reset(self):
        self.fallen_off_position = None  # Reset the fallen off position

        while True:
            start = np.random.uniform(-self.radius, self.radius, size=2)
            if np.linalg.norm(start) <= self.radius and not is_inside_obstacle(start, self):
                break

        while True:
            goal = np.random.uniform(-self.radius, self.radius, size=2)
            if np.linalg.norm(goal) <= self.radius and not is_inside_obstacle(goal, self):
                break

        return start, goal

class World:
    def __init__(self,
                 world_bounds,
                 num_obstacles = 2,
                 max_speed=100,
                 obstacle_radius=0.12,
                 friction=1.0,
                 num_vertices=200,
                 obs_random_position=False,
                 given_centers = None,
                 given_obstacles = None):
        self.obstacle_radius = obstacle_radius
        self.space = pymunk.Space()
        self.max_speed = max_speed
        self.obstacles = []
        self.world_bounds = world_bounds
        self.friction = friction
        self.num_vertices = num_vertices
        self.centers = []
        self.obs_random_position = obs_random_position
        self.given_centers = given_centers
        self.given_obstacles = given_obstacles

        xmin, xmax, ymin, ymax = self.world_bounds

        if self.given_obstacles is not None: 
            self.add_obstacle(self.given_obstacles)
                    
        elif self.given_centers is not None:
            self.centers = given_centers
            for (center_x, center_y) in self.given_centers:
                obstacle = self.generate_random_obstacle(center=(center_x, center_y))
                self.add_obstacle(obstacle)
        elif obs_random_position:
            # Random placement of obstacles
            self.generate_random_obstacle_positions(num_obstacles)
        else:
            # Original symmetric placement of obstacles
            num_per_row = int(np.sqrt(num_obstacles))
            dx = (xmax - xmin) / (num_per_row + 1)
            dy = (ymax - ymin) / (num_per_row + 1)
            for i in range(num_per_row):
                for j in range(num_per_row):
                    center_x = xmin + dx * (i + 1)
                    center_y = ymin + dy * (j + 1)
                    self.centers.append((center_x, center_y))
                    obstacle = self.generate_random_obstacle(center=(center_x, center_y))
                    self.add_obstacle(obstacle)

    def generate_random_obstacle_positions(self, num_obstacles):
        xmin, xmax, ymin, ymax = self.world_bounds
        x_min = xmin + self.obstacle_radius
        x_max = xmax - self.obstacle_radius
        y_min = ymin + self.obstacle_radius
        y_max = ymax - self.obstacle_radius

        max_attempts = 1000
        for _ in range(num_obstacles):
            for attempt in range(max_attempts):
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                center = (x, y)
                acceptable = True
                for c in self.centers:
                    distance = np.linalg.norm(np.array(center) - np.array(c))
                    if distance < 2 * self.obstacle_radius:
                        acceptable = False
                        break
                if acceptable:
                    self.centers.append(center)
                    obstacle = self.generate_random_obstacle(center=center)
                    self.add_obstacle(obstacle)
                    break
            else:
                raise ValueError("Unable to place all obstacles without overlap. Try reducing obstacle radius or number of obstacles.")

    def add_obstacle(self, obstacle):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Poly(body, obstacle)
        self.space.add(body, shape)
        self.obstacles.append((body, shape))

    def generate_random_obstacle(self, center):
        num_vertices = self.num_vertices
        obstacle = [self.generate_random_vertex_around(center, self.obstacle_radius) for _ in range(num_vertices)]
        return obstacle

    def generate_random_vertex_around(self, center, radius):
        angle = np.random.uniform(0, 2 * np.pi)
        distance = radius
        x = center[0] + np.cos(angle) * distance
        y = center[1] + np.sin(angle) * distance
        return (x, y)

    def actual_dynamics(self, position, desired_velocity, dt):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = tuple(position)  # Convert to tuple
        body.velocity = tuple(desired_velocity * dt)  # Convert to tuple

        self.space.add(body)

        collision_detected = False
        for obstacle in self.obstacles:
            query_info = self.space.segment_query_first(body.position, body.position + body.velocity, 0, pymunk.ShapeFilter())
            if query_info:
                collision_detected = True  # collision detected
                collision_normal = query_info.normal  # get the normal vector at the point of collision
                break

        # Initialize collision_detected and min_distance as empty lists
        collision_detected_all_obs = np.zeros(len(self.obstacles), dtype=bool)
        min_distance_all_obs = np.full(len(self.obstacles), float('inf'))
        for i, obstacle in enumerate(self.obstacles):
            query_info = obstacle[1].point_query(pymunk.Vec2d(position[0], position[1]))
            if query_info.distance < min_distance_all_obs[i]:
                min_distance_all_obs[i] = query_info.distance
                
            query_info = obstacle[1].segment_query(body.position, body.position + body.velocity)
            if query_info.shape is not None:
                collision_detected_all_obs[i] = True
                
        self.space.remove(body)

        # Calculate the new velocity
        speed = np.linalg.norm(desired_velocity)
        if collision_detected:
            # Decompose the velocity into components parallel and perpendicular to the normal vector
            velocity_perpendicular = np.dot(desired_velocity, collision_normal) * collision_normal
            velocity_parallel = (desired_velocity - velocity_perpendicular) * self.friction

            # Cancel the perpendicular component of the velocity, but leave the parallel component unchanged
            actual_velocity = velocity_parallel
        elif speed > self.max_speed:
            actual_velocity = (desired_velocity / speed) * self.max_speed
        else:
            actual_velocity = desired_velocity

        new_position = position + actual_velocity * dt
        # Clip the new position to the world boundaries
        world_bounds = np.array([self.world_bounds[0], self.world_bounds[1], self.world_bounds[2], self.world_bounds[3]])

        new_position = np.clip(new_position, world_bounds[0::2], world_bounds[1::2])

        return new_position, actual_velocity, collision_detected_all_obs, None, min_distance_all_obs




def is_inside_obstacle(position, world):
    position = tuple(position)  # Convert numpy array to tuple
    for obstacle in world.obstacles:
        if obstacle[1].point_query(position).distance <= 0:
            return True
    return False

def generate_random_positions(world, world_bounds=(-1, 1, -1, 1), orientation=None, circular=False):
    while True:
        if circular:
            start = np.random.uniform(-world.radius, world.radius, size=2)
            if np.linalg.norm(start) > world.radius:
                continue  # Regenerate if the point is outside the circle
        else:
            start = np.random.uniform(low=[world_bounds[0], world_bounds[2]], high=[world_bounds[1], world_bounds[3]])
        
        if orientation == 'orthogonal':
            # Randomly choose whether to align x or y
            align_x = np.random.choice([True, False])
            if align_x:
                # Align x, choose random y within bounds
                goal = [start[0], np.random.uniform(world_bounds[2], world_bounds[3])]
            else:
                # Align y, choose random x within bounds
                goal = [np.random.uniform(world_bounds[0], world_bounds[1]), start[1]]
        elif orientation == 'diagonal':
            # Choose a random displacement within the bounds, ensuring it's not too large that the goal would be out of bounds
            displacement = np.random.uniform(max(world_bounds[0]-start[0], world_bounds[2]-start[1]), min(world_bounds[1]-start[0], world_bounds[3]-start[1]))
            # Choose a random direction for the displacement
            direction = np.random.choice([1, -1])
            goal = [start[0] + direction * displacement, start[1] + displacement]
        elif orientation == 'discrete_distance':
            # Choose a random distance which is a multiple of 1/5 of the world's length
            distance = (world_bounds[1]-world_bounds[0]) * np.random.choice([0.1, 0.2, 0.4, 0.6, 0.65, 0.9])
            # Choose a random direction
            angle = np.random.uniform(0, 2*np.pi)
            # Calculate goal
            goal = [start[0] + distance * np.cos(angle), start[1] + distance * np.sin(angle)]
            # Check if the goal is inside the world boundaries
            if goal[0] < world_bounds[0] or goal[0] > world_bounds[1] or goal[1] < world_bounds[2] or goal[1] > world_bounds[3]:
                continue
        elif orientation == 'discrete':
            # Choose a random distance which is a multiple of a certain fraction of the world's length
            distance = np.random.uniform(0, 0.6)
            # Choose one of the 8 possible angles (0, 45, 90, 135, 180, 225, 270, 315 degrees)
            angle = np.random.choice([i * np.pi / 4 for i in range(8)])
            goal = [start[0] + distance * np.cos(angle), start[1] + distance * np.sin(angle)]
            if goal[0] < world_bounds[0] or goal[0] > world_bounds[1] or goal[1] < world_bounds[2] or goal[1] > world_bounds[3]:
                continue
        else:
            if circular:
                while True:
                    goal = np.random.uniform(-world.radius, world.radius, size=2)
                    if np.linalg.norm(goal) <= world.radius:
                        break
            else:
                goal = np.random.uniform(low=[world_bounds[0], world_bounds[2]], high=[world_bounds[1], world_bounds[3]])
        
        if not is_inside_obstacle(start, world) and not is_inside_obstacle(goal, world):
                return start, goal



class ContinuousActionWorld(World):
    def __init__(self, num_obstacles, max_speed, world_bounds, step_size=0.02,obstacle_radius=0.06):
        super().__init__(num_obstacles, max_speed, world_bounds,obstacle_radius)
        self.step_size = step_size  # the distance the agent moves in one step

    def action_to_velocity(self, action):
        # Convert the action (angle) to a unit vector
        u = np.array([np.cos(action), np.sin(action)])
        # Multiply the unit vector with the step size to get the desired velocity
        desired_velocity = u * self.step_size
        return desired_velocity

    def step(self, position, action):
        # Convert the action (angle) to desired_velocity
        desired_velocity = self.action_to_velocity(action)
        # Compute the actual dynamics using the desired velocity
        new_position, actual_velocity, collision_detected_all_obs, _, min_distance_all_obs = self.actual_dynamics(position, desired_velocity, self.step_size)
        return new_position, actual_velocity, collision_detected_all_obs, min_distance_all_obs
    
    def reset(self):
        start, goal = generate_random_positions(self, self.world_bounds)
        return start, goal
    

if __name__ == '__main__':
    # Create environment instance
    env = CircularWorld(num_obstacles=0, max_speed=0.5, radius=1, wall_present=True, obstacle_radius=0.0)

    # Initialize start and goal positions
    start, goal = env.reset()

    # Define number of steps for the demonstration
    num_steps = 200

    # List to store all positions
    position_track = [start]

    # Simulate the agent's movement in the environment
    current_position = start
    for _ in range(num_steps):
        # Sample a random action (velocity direction angle between 0 and 2π)
        angle = np.random.uniform(0, 2 * np.pi)
        desired_velocity = np.array([np.cos(angle), np.sin(angle)]) * env.max_speed
        new_position, actual_velocity, collision_detected, collision_detected_all_obs, min_distance_all_obs = env.actual_dynamics(current_position, desired_velocity, env.max_speed)
        position_track.append(new_position)
        current_position = new_position

    position_track = np.array(position_track)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # Plot obstacles
    for body, obstacle in env.obstacles:
        vertices = np.array([body.local_to_world(v) for v in obstacle.get_vertices()])
        vertices = np.vstack((vertices, vertices[0]))  # Close the polygon by adding the first point at the end
        plt.plot(*vertices.T, color='black')
        plt.fill(*vertices.T, color='grey', alpha=0.5)  # Fill the polygon to make the obstacle solid

    # Plot the trajectory
    for i in range(len(position_track) - 2):
        plt.plot(position_track[i:i + 2, 0], position_track[i:i + 2, 1], color='blue', alpha=0.6)
    plt.scatter(*goal, color='red', s=100)
    plt.title("Agent's Trajectory in CircularWorld")
    plt.show()
