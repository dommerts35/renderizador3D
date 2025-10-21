import pygame
import moderngl
import numpy as np
import math

class Renderer3D:
    def __init__(self, width=800, height=600):

        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Renderizador 3D con Iluminación Phong")
        self.ctx = moderngl.create_context()
        
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        self.camera_pos = np.array([0.0, 2.0, 8.0], dtype=np.float32)
        self.camera_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.camera_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        self.yaw = -90.0
        self.pitch = 0.0
        
        self.camera_speed = 0.1
        self.mouse_sensitivity = 0.1
        
        self.keys_pressed = {
            pygame.K_w: False,
            pygame.K_s: False,
            pygame.K_a: False,
            pygame.K_d: False,
            pygame.K_SPACE: False,
            pygame.K_LSHIFT: False,
            pygame.K_1: False,
            pygame.K_2: False,
            pygame.K_3: False
        }
        
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        
        self.light_pos = np.array([3.0, 5.0, 2.0], dtype=np.float32)
        self.light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.ambient_strength = 0.2
        self.specular_strength = 0.5
        self.shininess = 32
        
        self.materials = {
            'cube': {
                'ambient': np.array([1.0, 0.5, 0.31], dtype=np.float32),
                'diffuse': np.array([1.0, 0.5, 0.31], dtype=np.float32),
                'specular': np.array([0.5, 0.5, 0.5], dtype=np.float32),
            },
            'pyramid': {
                'ambient': np.array([0.0, 0.5, 0.8], dtype=np.float32),
                'diffuse': np.array([0.0, 0.5, 0.8], dtype=np.float32),
                'specular': np.array([0.7, 0.7, 0.7], dtype=np.float32),
            },
            'floor': {
                'ambient': np.array([0.3, 0.3, 0.3], dtype=np.float32),
                'diffuse': np.array([0.4, 0.4, 0.4], dtype=np.float32),
                'specular': np.array([0.2, 0.2, 0.2], dtype=np.float32),
            }
        }
        
        self.create_shaders()
        
        self.create_cube()
        self.create_pyramid()
        self.create_floor()
        self.create_light_source() 
        
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        
        self.time = 0.0
        
    def create_shaders(self):
        vertex_shader = """
        #version 330 core
        
        layout (location = 0) in vec3 in_position;
        layout (location = 1) in vec3 in_normal;
        layout (location = 2) in vec3 in_color;
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec3 Color;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        void main() {
            FragPos = vec3(model * vec4(in_position, 1.0));
            Normal = mat3(transpose(inverse(model))) * in_normal;
            Color = in_color;
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330 core
        
        in vec3 FragPos;
        in vec3 Normal;
        in vec3 Color;
        
        out vec4 out_color;
        
        uniform vec3 viewPos;
        uniform vec3 lightPos;
        uniform vec3 lightColor;
        uniform vec3 materialAmbient;
        uniform vec3 materialDiffuse;
        uniform vec3 materialSpecular;
        uniform float ambientStrength;
        uniform float specularStrength;
        uniform float shininess;
        
        void main() {
            vec3 ambient = ambientStrength * lightColor * materialAmbient;
            
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor * materialDiffuse;
            
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
            vec3 specular = specularStrength * spec * lightColor * materialSpecular;
            
            vec3 result = (ambient + diffuse + specular) * Color;
            out_color = vec4(result, 1.0);
        }
        """
        
        self.prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
        self.model_uniform = self.prog['model']
        self.view_uniform = self.prog['view']
        self.projection_uniform = self.prog['projection']
        self.view_pos_uniform = self.prog['viewPos']
        self.light_pos_uniform = self.prog['lightPos']
        self.light_color_uniform = self.prog['lightColor']
        self.material_ambient_uniform = self.prog['materialAmbient']
        self.material_diffuse_uniform = self.prog['materialDiffuse']
        self.material_specular_uniform = self.prog['materialSpecular']
        self.ambient_strength_uniform = self.prog['ambientStrength']
        self.specular_strength_uniform = self.prog['specularStrength']
        self.shininess_uniform = self.prog['shininess']
    
    def create_cube(self):
        vertices = [
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 0.0, 0.0,
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 0.0, 0.0,
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 0.0, 0.0,
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 0.0, 0.0,
            
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 1.0, 0.0,
             0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 1.0, 0.0,
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 1.0, 0.0,
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 1.0, 0.0,
            
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 0.0, 1.0,
            -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  0.0, 0.0, 1.0,
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  0.0, 0.0, 1.0,
            -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 0.0, 1.0,
            
             0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  1.0, 1.0, 0.0,
             0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  1.0, 1.0, 0.0,
             0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0, 1.0, 0.0,
             0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  1.0, 1.0, 0.0,
            
            -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 0.0, 1.0,
             0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 0.0, 1.0,
             0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0, 0.0, 1.0,
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0, 0.0, 1.0,
            
            -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0, 1.0, 1.0,
             0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0, 1.0, 1.0,
             0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 1.0, 1.0,
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 1.0, 1.0,
        ]
        
        indices = [
            0, 1, 2, 2, 3, 0,    
            4, 5, 6, 6, 7, 4,    
            8, 9, 10, 10, 11, 8, 
            12, 13, 14, 14, 15, 12, 
            16, 17, 18, 18, 19, 16, 
            20, 21, 22, 22, 23, 20 
        ]
        
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        self.cube_vbo = self.ctx.buffer(vertices.tobytes())
        self.cube_ebo = self.ctx.buffer(indices.tobytes())
        
        self.cube_vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.cube_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')
            ],
            self.cube_ebo
        )
        
        self.cube_num_indices = len(indices)
    
    def create_pyramid(self):
        vertices = [
            -1.0, -1.0, -1.0,  0.0, -1.0,  0.0,  0.0, 0.0, 1.0,
             1.0, -1.0, -1.0,  0.0, -1.0,  0.0,  0.0, 1.0, 0.0,
             0.0, -1.0,  1.0,  0.0, -1.0,  0.0,  1.0, 0.0, 0.0,
            
            -1.0, -1.0, -1.0,  -0.816, 0.333, -0.471,  0.0, 0.0, 1.0,
             0.0, -1.0,  1.0,  -0.816, 0.333, -0.471,  1.0, 0.0, 0.0,
             0.0,  1.0,  0.0,  -0.816, 0.333, -0.471,  1.0, 1.0, 0.0,
            
             0.0, -1.0,  1.0,  0.0, 0.333, 0.943,  1.0, 0.0, 0.0,
             1.0, -1.0, -1.0,  0.0, 0.333, 0.943,  0.0, 1.0, 0.0,
             0.0,  1.0,  0.0,  0.0, 0.333, 0.943,  1.0, 1.0, 0.0,
            
             1.0, -1.0, -1.0,  0.816, 0.333, -0.471,  0.0, 1.0, 0.0,
            -1.0, -1.0, -1.0,  0.816, 0.333, -0.471,  0.0, 0.0, 1.0,
             0.0,  1.0,  0.0,  0.816, 0.333, -0.471,  1.0, 1.0, 0.0,
        ]
        
        indices = list(range(12))
        
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        self.pyramid_vbo = self.ctx.buffer(vertices.tobytes())
        self.pyramid_ebo = self.ctx.buffer(indices.tobytes())
        
        self.pyramid_vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.pyramid_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')
            ],
            self.pyramid_ebo
        )
        
        self.pyramid_num_indices = len(indices)
    
    def create_floor(self):
        vertices = [
            -10.0, -2.0, -10.0,  0.0, 1.0, 0.0,  0.3, 0.3, 0.3,
             10.0, -2.0, -10.0,  0.0, 1.0, 0.0,  0.3, 0.3, 0.3,
             10.0, -2.0,  10.0,  0.0, 1.0, 0.0,  0.3, 0.3, 0.3,
            -10.0, -2.0,  10.0,  0.0, 1.0, 0.0,  0.3, 0.3, 0.3,
        ]
        
        indices = [0, 1, 2, 2, 3, 0]
        
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        self.floor_vbo = self.ctx.buffer(vertices.tobytes())
        self.floor_ebo = self.ctx.buffer(indices.tobytes())
        
        self.floor_vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.floor_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')
            ],
            self.floor_ebo
        )
        
        self.floor_num_indices = len(indices)
    
    def create_light_source(self):
        light_vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 in_position;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * model * vec4(in_position, 1.0);
        }
        """
        
        light_fragment_shader = """
        #version 330 core
        out vec4 out_color;
        uniform vec3 lightColor;
        void main() {
            out_color = vec4(lightColor, 1.0);
        }
        """
        
        self.light_prog = self.ctx.program(
            vertex_shader=light_vertex_shader,
            fragment_shader=light_fragment_shader
        )
        
        vertices = [
            -0.2, -0.2, -0.2,
             0.2, -0.2, -0.2,
             0.2,  0.2, -0.2,
            -0.2,  0.2, -0.2,
            -0.2, -0.2,  0.2,
             0.2, -0.2,  0.2,
             0.2,  0.2,  0.2,
            -0.2,  0.2,  0.2,
        ]
        
        indices = [
            0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1,
            5, 4, 7, 7, 6, 5, 4, 0, 3, 3, 7, 4,
            3, 2, 6, 6, 7, 3, 4, 5, 1, 1, 0, 4
        ]
        
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        self.light_vbo = self.ctx.buffer(vertices.tobytes())
        self.light_ebo = self.ctx.buffer(indices.tobytes())
        
        self.light_vao = self.ctx.vertex_array(
            self.light_prog,
            [
                (self.light_vbo, '3f', 'in_position')
            ],
            self.light_ebo
        )
        
        self.light_num_indices = len(indices)
        self.light_color_uniform_light = self.light_prog['lightColor']
        self.light_model_uniform = self.light_prog['model']
        self.light_view_uniform = self.light_prog['view']
        self.light_projection_uniform = self.light_prog['projection']
    
    def update_camera_vectors(self):
        front = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ], dtype=np.float32)
        
        self.camera_front = front / np.linalg.norm(front)
        self.camera_right = np.cross(self.camera_front, np.array([0.0, 1.0, 0.0]))
        self.camera_right = self.camera_right / np.linalg.norm(self.camera_right)
        self.camera_up = np.cross(self.camera_right, self.camera_front)
    
    def process_mouse_movement(self, xoffset, yoffset):
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity
        
        self.yaw += xoffset
        self.pitch += yoffset
        
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0
        
        self.update_camera_vectors()
    
    def process_keyboard(self):
        velocity = self.camera_speed
        
        if self.keys_pressed[pygame.K_w]:
            self.camera_pos += self.camera_front * velocity
        if self.keys_pressed[pygame.K_s]:
            self.camera_pos -= self.camera_front * velocity
        if self.keys_pressed[pygame.K_a]:
            self.camera_pos -= self.camera_right * velocity
        if self.keys_pressed[pygame.K_d]:
            self.camera_pos += self.camera_right * velocity
        if self.keys_pressed[pygame.K_SPACE]:
            self.camera_pos += self.camera_up * velocity
        if self.keys_pressed[pygame.K_LSHIFT]:
            self.camera_pos -= self.camera_up * velocity
        
        if self.keys_pressed[pygame.K_1]:
            self.light_pos[0] += 0.1
        if self.keys_pressed[pygame.K_2]:
            self.light_pos[1] += 0.1
        if self.keys_pressed[pygame.K_3]:
            self.light_pos[2] += 0.1
    
    def get_view_matrix(self):
        return self.look_at(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)
    
    def look_at(self, position, target, up):
        zaxis = (position - target) / np.linalg.norm(position - target)
        xaxis = np.cross(up, zaxis) / np.linalg.norm(np.cross(up, zaxis))
        yaxis = np.cross(zaxis, xaxis)
        
        translation = np.identity(4)
        translation[0:3, 3] = -position
        
        rotation = np.identity(4)
        rotation[0, 0:3] = xaxis
        rotation[1, 0:3] = yaxis
        rotation[2, 0:3] = zaxis
        
        return np.dot(rotation, translation)
    
    def get_projection_matrix(self):
        fov = math.radians(60.0)
        aspect = self.width / self.height
        near = 0.1
        far = 100.0
        
        f = 1.0 / math.tan(fov / 2.0)
        projection = np.zeros((4, 4))
        projection[0, 0] = f / aspect
        projection[1, 1] = f
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = (2.0 * far * near) / (near - far)
        projection[3, 2] = -1.0
        
        return projection
    
    def get_model_matrix(self, position, rotation_x=0.0, rotation_y=0.0, rotation_z=0.0, scale=1.0):
        model = np.identity(4)
        model[0, 0] = scale
        model[1, 1] = scale
        model[2, 2] = scale
        model[0, 3] = position[0]
        model[1, 3] = position[1]
        model[2, 3] = position[2]
        
        if rotation_x != 0:
            rx = np.identity(4)
            rx[1, 1] = math.cos(rotation_x)
            rx[1, 2] = -math.sin(rotation_x)
            rx[2, 1] = math.sin(rotation_x)
            rx[2, 2] = math.cos(rotation_x)
            model = np.dot(model, rx)
        
        if rotation_y != 0:
            ry = np.identity(4)
            ry[0, 0] = math.cos(rotation_y)
            ry[0, 2] = math.sin(rotation_y)
            ry[2, 0] = -math.sin(rotation_y)
            ry[2, 2] = math.cos(rotation_y)
            model = np.dot(model, ry)
        
        if rotation_z != 0:
            rz = np.identity(4)
            rz[0, 0] = math.cos(rotation_z)
            rz[0, 1] = -math.sin(rotation_z)
            rz[1, 0] = math.sin(rotation_z)
            rz[1, 1] = math.cos(rotation_z)
            model = np.dot(model, rz)
        
        return model
    
    def render(self):
        self.ctx.clear(0.0, 0.1, 0.2)
        
        view = self.get_view_matrix()
        projection = self.get_projection_matrix()
        
        self.view_uniform.write(view.T.astype('f4').tobytes())
        self.projection_uniform.write(projection.T.astype('f4').tobytes())
        self.view_pos_uniform.write(self.camera_pos.astype('f4').tobytes())
        self.light_pos_uniform.write(self.light_pos.astype('f4').tobytes())
        self.light_color_uniform.write(self.light_color.astype('f4').tobytes())
        self.ambient_strength_uniform.value = self.ambient_strength
        self.specular_strength_uniform.value = self.specular_strength
        self.shininess_uniform.value = self.shininess
        
        floor_model = self.get_model_matrix([0.0, 0.0, 0.0])
        self.model_uniform.write(floor_model.T.astype('f4').tobytes())
        self.material_ambient_uniform.write(self.materials['floor']['ambient'].astype('f4').tobytes())
        self.material_diffuse_uniform.write(self.materials['floor']['diffuse'].astype('f4').tobytes())
        self.material_specular_uniform.write(self.materials['floor']['specular'].astype('f4').tobytes())
        self.floor_vao.render()
        
        cube_model = self.get_model_matrix(
            [-2.0, 0.0, 0.0],
            self.rotation_x,
            self.rotation_y,
            self.rotation_x * 0.3,
            0.8
        )
        self.model_uniform.write(cube_model.T.astype('f4').tobytes())
        self.material_ambient_uniform.write(self.materials['cube']['ambient'].astype('f4').tobytes())
        self.material_diffuse_uniform.write(self.materials['cube']['diffuse'].astype('f4').tobytes())
        self.material_specular_uniform.write(self.materials['cube']['specular'].astype('f4').tobytes())
        self.cube_vao.render()
        
        pyramid_model = self.get_model_matrix(
            [2.0, 0.0, 0.0],
            self.rotation_y * 0.7,
            self.rotation_x * 0.5,
            self.rotation_y * 0.9,
            1.2
        )
        self.model_uniform.write(pyramid_model.T.astype('f4').tobytes())
        self.material_ambient_uniform.write(self.materials['pyramid']['ambient'].astype('f4').tobytes())
        self.material_diffuse_uniform.write(self.materials['pyramid']['diffuse'].astype('f4').tobytes())
        self.material_specular_uniform.write(self.materials['pyramid']['specular'].astype('f4').tobytes())
        self.pyramid_vao.render(moderngl.TRIANGLES, self.pyramid_num_indices)
        
        self.light_view_uniform.write(view.T.astype('f4').tobytes())
        self.light_projection_uniform.write(projection.T.astype('f4').tobytes())
        self.light_color_uniform_light.write(self.light_color.astype('f4').tobytes())
        
        light_source_model = self.get_model_matrix(self.light_pos, scale=0.3)
        self.light_model_uniform.write(light_source_model.T.astype('f4').tobytes())
        self.light_vao.render()
        
        pygame.display.flip()
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            current_time = pygame.time.get_ticks() / 1000.0
            delta_time = current_time - self.time
            self.time = current_time

            self.light_pos[0] = 3.0 + math.sin(current_time) * 2.0
            self.light_pos[1] = 5.0 + math.cos(current_time * 0.5) * 1.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in self.keys_pressed:
                        self.keys_pressed[event.key] = True
                elif event.type == pygame.KEYUP:
                    if event.key in self.keys_pressed:
                        self.keys_pressed[event.key] = False
                elif event.type == pygame.MOUSEMOTION:
                    x, y = event.pos
                    xoffset = x - self.width // 2
                    yoffset = self.height // 2 - y
                    self.process_mouse_movement(xoffset, yoffset)
                    pygame.mouse.set_pos((self.width // 2, self.height // 2))
            
            self.process_keyboard()
            
            self.rotation_x += 0.01
            self.rotation_y += 0.015
            
            self.render()
            
            pygame.display.set_caption(
                f"Iluminación Phong | FPS: {clock.get_fps():.1f} | "
                f"Pos: ({self.camera_pos[0]:.1f}, {self.camera_pos[1]:.1f}, {self.camera_pos[2]:.1f}) | "
                f"Luz: ({self.light_pos[0]:.1f}, {self.light_pos[1]:.1f}, {self.light_pos[2]:.1f}) | "
                f"WASD: Moverse | 1/2/3: Mover luz | ESC: Salir"
            )
            
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    renderer = Renderer3D()

    renderer.run()
