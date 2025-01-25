package main

import "base:runtime"
import "base:intrinsics"
import t "core:time"
import "core:fmt"
import "core:os"
import "core:math"
import "core:math/linalg"
import "core:math/ease"
import "core:mem"
import "core:slice"
import "core:strings"
import time "core:time"
import rand "core:math/rand"
import json "core:encoding/json"

import sapp "sokol/app"
import sg "sokol/gfx"
import sglue "sokol/glue"
import slog "sokol/log"

import stbi "vendor:stb/image"
import stbrp "vendor:stb/rect_pack"
import stbtt "vendor:stb/truetype"

app_state: struct {
	pass_action: sg.Pass_Action,
	pip: sg.Pipeline,
	bind: sg.Bindings,
	game: Game_State,
	input_state: Input_State,
}

window_w :: 1280
window_h :: 720

main :: proc() {
	sapp.run({
		init_cb = init,
		frame_cb = frame,
		cleanup_cb = cleanup,
		event_cb = event,
		width = window_w,
		height = window_h,
		window_title = "Dummy",
		icon = { sokol_default = true },
		logger = { func = slog.func },
		win32_console_attach = false,
	})
}

init :: proc "c" () {
	using linalg, fmt
	context = runtime.default_context()
	sapp.toggle_fullscreen()

	init_time = t.now()

	sg.setup({
		environment = sglue.environment(),
		logger = { func = slog.func },
		d3d11_shader_debugging = ODIN_DEBUG,
	})

	init_images()
	init_fonts()

	// :init
	gs = &app_state.game
    gs.ui_hot_reload = init_ui_hot_reload()
    gs.ui_config = gs.ui_hot_reload.config

    gs.skills_system = init_skills_system()
    gs.quests_system = init_quests_system()
    gs.upgrades_system = init_upgrades_system()

    for &e, kind in entity_data {
        setup_entity(&e, kind)
    }

	app_state.bind.vertex_buffers[0] = sg.make_buffer({
		usage = .DYNAMIC,
		size = size_of(Quad) * len(draw_frame.quads),
	})

	index_buffer_count :: MAX_QUADS*6
	indices : [index_buffer_count]u16;
	i := 0;
	for i < index_buffer_count {
		indices[i + 0] = auto_cast ((i/6)*4 + 0)
		indices[i + 1] = auto_cast ((i/6)*4 + 1)
		indices[i + 2] = auto_cast ((i/6)*4 + 2)
		indices[i + 3] = auto_cast ((i/6)*4 + 0)
		indices[i + 4] = auto_cast ((i/6)*4 + 2)
		indices[i + 5] = auto_cast ((i/6)*4 + 3)
		i += 6;
	}
	app_state.bind.index_buffer = sg.make_buffer({
		type = .INDEXBUFFER,
		data = { ptr = &indices, size = size_of(indices) },
	})

	app_state.bind.samplers[SMP_default_sampler] = sg.make_sampler({})

	pipeline_desc : sg.Pipeline_Desc = {
		shader = sg.make_shader(quad_shader_desc(sg.query_backend())),
		index_type = .UINT16,
		layout = {
			attrs = {
				ATTR_quad_position = { format = .FLOAT2 },
				ATTR_quad_color0 = { format = .FLOAT4 },
				ATTR_quad_uv0 = { format = .FLOAT2 },
				ATTR_quad_bytes0 = { format = .UBYTE4N },
				ATTR_quad_color_override0 = { format = .FLOAT4 }
			},
		}
	}
	blend_state : sg.Blend_State = {
		enabled = true,
		src_factor_rgb = .SRC_ALPHA,
		dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
		op_rgb = .ADD,
		src_factor_alpha = .ONE,
		dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
		op_alpha = .ADD,
	}
	pipeline_desc.colors[0] = { blend = blend_state }
	app_state.pip = sg.make_pipeline(pipeline_desc)

	app_state.pass_action = {
		colors = {
			0 = { load_action = .CLEAR, clear_value = { 0, 0, 0, 1 }},
		},
	}
}

//
// :frame
frame :: proc "c" () {
	using runtime, linalg
	context = runtime.default_context()

    width := sapp.width()
    height := sapp.height()
    target_ratio := f32(game_res_w) / f32(game_res_h)
    window_ratio := f32(width) / f32(height)

    viewport_x, viewport_y : i32
    viewport_w, viewport_h : i32

    if window_ratio > target_ratio {
        viewport_h = auto_cast height
        viewport_w = i32(f32(height) * target_ratio)
        viewport_x = (auto_cast width - viewport_w) / 2
        viewport_y = 0
    } else {
        viewport_w = auto_cast width
        viewport_h = i32(f32(width) / target_ratio)
        viewport_x = 0
        viewport_y = (auto_cast height - viewport_h) / 2
    }

	draw_frame.reset = {}

	update()
	render()

	dt := sapp.frame_duration()

	app_state.bind.images[IMG_tex0] = atlas.sg_image
	app_state.bind.images[IMG_tex1] = images[font.img_id].sg_img

	verts: Raw_Slice
	verts.len = draw_frame.quad_count
	verts.data = &draw_frame.quads[0]

	_v := transmute([]Quad)verts

	slice.stable_sort_by_cmp(_v, proc(a, b: Quad) -> slice.Ordering{
		return slice.cmp(a[0].z_layer, b[0].z_layer)
	})

	sg.update_buffer(
		app_state.bind.vertex_buffers[0],
		{ ptr = &draw_frame.quads[0], size = size_of(Quad) * len(draw_frame.quads) }
	)
	sg.begin_pass({ action = app_state.pass_action, swapchain = sglue.swapchain() })
	sg.apply_pipeline(app_state.pip)
	sg.apply_bindings(app_state.bind)
	sg.draw(0, 6*draw_frame.quad_count, 1)
	sg.end_pass()
	sg.commit()

	reset_input_state_for_next_frame(&app_state.input_state)
    free_all(context.temp_allocator)
}

cleanup :: proc "c" () {
	context = runtime.default_context()
	sg.shutdown()
}

//
// :UTILS

DEFAULT_UV :: v4{0, 0, 1, 1}
Vector2 :: [2]f32
Vector3 :: [3]f32
Vector4 :: [4]f32
v2 :: Vector2
v3 :: Vector3
v4 :: Vector4
Matrix4 :: linalg.Matrix4f32;

COLOR_WHITE :: Vector4 {1,1,1,1}
COLOR_BLACK :: Vector4{0,0,0,1}
COLOR_RED :: Vector4{1,0,0,1}

loggie :: fmt.println
log_error :: fmt.println
log_warning :: fmt.println

init_time: t.Time;
seconds_since_init :: proc() -> f64 {
	using t
	if init_time._nsec == 0 {
		log_error("invalid time")
		return 0
	}
	return duration_seconds(since(init_time))
}

xform_translate :: proc(pos: Vector2) -> Matrix4 {
	return linalg.matrix4_translate(v3{pos.x, pos.y, 0})
}
xform_rotate :: proc(angle: f32) -> Matrix4 {
	return linalg.matrix4_rotate(math.to_radians(angle), v3{0,0,1})
}
xform_scale :: proc(scale: Vector2) -> Matrix4 {
	return linalg.matrix4_scale(v3{scale.x, scale.y, 1});
}

sign :: proc(x: f32) -> f32 {
    if x < 0 do return -1
    if x > 0 do return 1
    return 0
}

Pivot :: enum {
	bottom_left,
	bottom_center,
	bottom_right,
	center_left,
	center_center,
	center_right,
	top_left,
	top_center,
	top_right,
}

scale_from_pivot :: proc(pivot: Pivot) -> Vector2 {
	switch pivot {
		case .bottom_left: return v2{0.0, 0.0}
		case .bottom_center: return v2{0.5, 0.0}
		case .bottom_right: return v2{1.0, 0.0}
		case .center_left: return v2{0.0, 0.5}
		case .center_center: return v2{0.5, 0.5}
		case .center_right: return v2{1.0, 0.5}
		case .top_center: return v2{0.5, 1.0}
		case .top_left: return v2{0.0, 1.0}
		case .top_right: return v2{1.0, 1.0}
	}
	return {};
}

sine_breathe :: proc(p: $T) -> T where intrinsics.type_is_float(T) {
	return (math.sin((p - .25) * 2.0 * math.PI) / 2.0) + 0.5
}

animate_to_target_f32 :: proc(current: ^f32, target: f32, dt: f32, rate: f32 = 10.0) {
    diff := target - current^
    current^ += diff * min(1.0, dt * rate)
}

interpolate :: proc(a, b: $T, t: f32) -> T where intrinsics.type_is_float(T) {
    return a + (b - a) * t
}

// UI Utilities
get_skill_button_pos :: proc() -> Vector2 {
    return Vector2{0, game_res_h * 0.45}
}

get_menu_position :: proc(alpha: f32) -> Vector2 {
    base_pos := Vector2{0, game_res_h * 0.3}
    offset := Vector2{0, UI.MENU_PADDING}
    return base_pos + offset * alpha
}

is_point_in_rect :: proc(point, pos, size: Vector2) -> bool {
    half_size := size * 0.5
    min := pos - half_size
    max := pos + half_size

    return point.x >= min.x && point.x <= max.x &&
           point.y >= min.y && point.y <= max.y
}

ray_circle_intersection :: proc(ray_start, ray_end, circle_center: Vector2, circle_radius: f32) -> bool {
    ray_dir := ray_end - ray_start
    ray_length := linalg.length(ray_dir)

    if ray_length == 0 {
        return linalg.length2(ray_start - circle_center) <= circle_radius * circle_radius
    }

    ray_dir = ray_dir / ray_length

    start_to_center := circle_center - ray_start

    proj_length := linalg.dot(start_to_center, ray_dir)

    if proj_length < 0 || proj_length > ray_length {
        return false
    }

    closest_point := ray_start + ray_dir * proj_length

    closest_dist_sq := linalg.length2(closest_point - circle_center)
    return closest_dist_sq <= circle_radius * circle_radius
}

circle_collide :: proc(pos1, pos2: Vector2, radius1, radius2: f32) -> bool {
    diff := pos2 - pos1
    dist_sq := linalg.length2(diff)
    combined_radius := radius1 + radius2
    return dist_sq <= combined_radius * combined_radius
}

//
// :RENDER STUFF

draw_sprite :: proc(pos: Vector2, img_id: Image_Id, pivot:= Pivot.bottom_left, xform := Matrix4(1), color_override:= v4{0,0,0,0}, z_layer := ZLayer.nil) {
	image := images[img_id]
	size := v2{auto_cast image.width, auto_cast image.height}

	RES :: 256
	p_scale := f32(RES) / max(f32(image.width), f32(image.height))

	xform0 := Matrix4(1)
	xform0 *= xform_translate(pos)
	xform0 *= xform
	xform0 *= xform_scale(v2{p_scale, p_scale})
	xform0 *= xform_translate(size * -scale_from_pivot(pivot))

	draw_rect_xform(xform0, size, img_id=img_id, color_override=color_override, z_layer=z_layer)
}

draw_sprite_1024 :: proc(pos: Vector2, size := v2{0,0} , img_id: Image_Id, pivot:= Pivot.bottom_left, xform := Matrix4(1), color_override:= v4{0,0,0,0}, z_layer := ZLayer.nil) {
	image := images[img_id]
	size := size

	RES :: 1024
	p_scale := f32(RES) / max(f32(image.width), f32(image.height))

	xform0 := Matrix4(1)
	xform0 *= xform_translate(pos)
	xform0 *= xform
	xform0 *= xform_scale(v2{p_scale, p_scale})
	xform0 *= xform_translate(size * -scale_from_pivot(pivot))

	draw_rect_xform(xform0, size, img_id=img_id, color_override=color_override, z_layer=z_layer)
}

fit_size_to_square :: proc(target_size: Vector2) -> Vector2{
    max_dim := max(target_size.x, target_size.y)
    return Vector2{max_dim, max_dim}
}

draw_sprite_with_size :: proc(pos: Vector2, size := v2{0,0} , img_id: Image_Id, pivot:= Pivot.bottom_left, xform := Matrix4(1), color_override:= v4{0,0,0,0}, z_layer := ZLayer.nil) {
	image := images[img_id]
	size := size

	RES :: 256
	p_scale := f32(RES) / max(f32(image.width), f32(image.height))

	xform0 := Matrix4(1)
	xform0 *= xform_translate(pos)
	xform0 *= xform
	xform0 *= xform_scale(v2{p_scale, p_scale})
	xform0 *= xform_translate(size * -scale_from_pivot(pivot))

	draw_rect_xform(xform0, size, img_id=img_id, color_override=color_override, z_layer=z_layer)
}

draw_nores_sprite_with_size :: proc(pos: Vector2, size: Vector2, img_id: Image_Id, pivot := Pivot.bottom_left, xform := Matrix4(1), color_override := v4{0,0,0,0}, z_layer := ZLayer.nil) {
    image := images[img_id]

    xform0 := Matrix4(1)
    xform0 *= xform_translate(pos)
    xform0 *= xform
    xform0 *= xform_translate(size * -scale_from_pivot(pivot))

    draw_rect_xform(xform0, size, img_id=img_id, color_override=color_override, z_layer=z_layer)
}

draw_sprite_in_rect :: proc(img_id: Image_Id, pos: Vector2, size: Vector2, xform := Matrix4(1), col := COLOR_WHITE, color_override := v4{0,0,0,0}, z_layer := ZLayer.nil){
    image := images[img_id]
    img_size := v2{auto_cast image.width, auto_cast image.height}

    pos0 := pos
    pos0.x += (size.x - img_size.x) * 0.5
    pos0.y += (size.y - img_size.y) * 0.5

    draw_rect_aabb(pos0, img_size, col = col, img_id = img_id, color_override = color_override, z_layer = z_layer)
}

draw_rect_aabb_actually :: proc(
	aabb: AABB,
	col: Vector4=COLOR_WHITE,
	uv: Vector4=DEFAULT_UV,
	img_id: Image_Id=.nil,
	color_override:= v4{0,0,0,0},
	z_layer:=ZLayer.nil,
) {
	xform := linalg.matrix4_translate(v3{aabb.x, aabb.y, 0})
	draw_rect_xform(xform, aabb_size(aabb), col, uv, img_id, color_override, z_layer=z_layer)
}

draw_rect_aabb :: proc(
	pos: Vector2,
	size: Vector2,
	col: Vector4=COLOR_WHITE,
	uv: Vector4=DEFAULT_UV,
	img_id: Image_Id=.nil,
	color_override:= v4{0,0,0,0},
	z_layer := ZLayer.nil,
) {
	xform := linalg.matrix4_translate(v3{pos.x, pos.y, 0})
	draw_rect_xform(xform, size, col, uv, img_id, color_override, z_layer=z_layer)
}

draw_rect_xform :: proc(
	xform: Matrix4,
	size: Vector2,
	col: Vector4=COLOR_WHITE,
	uv: Vector4=DEFAULT_UV,
	img_id: Image_Id=.nil,
	color_override:= v4{0,0,0,0},
	z_layer := ZLayer.nil,
) {
	draw_rect_projected(draw_frame.coord_space.proj * draw_frame.coord_space.camera * xform, size, col, uv, img_id, color_override, z_layer=z_layer)
}

Vertex :: struct {
	pos: Vector2,
	col: Vector4,
	uv: Vector2,
	tex_index: u8,
	z_layer: u8,
	_: [2]u8,
	color_override: Vector4,
}

Quad :: [4]Vertex;

MAX_QUADS :: 8192
MAX_VERTS :: MAX_QUADS * 4

Draw_Frame :: struct {
	quads: [MAX_QUADS]Quad,

    using reset: struct {
        coord_space: Coord_Space,
        quad_count: int,
        flip_v: bool,
        active_z_layer: ZLayer,
    }
}
draw_frame : Draw_Frame

ZLayer :: enum u8{
    nil,
    background,
    player,
    foreground,
    midground,
    ui,
    xp_bars,
    bow,
}

Coord_Space :: struct {
    proj: Matrix4,
    camera: Matrix4,
}

set_coord_space :: proc(coord: Coord_Space) {
	draw_frame.coord_space = coord
}

@(deferred_out=set_coord_space)
push_coord_space :: proc(coord: Coord_Space) -> Coord_Space {
    og := draw_frame.coord_space
    draw_frame.coord_space = coord
    return og
}

set_z_layer :: proc(zlayer: ZLayer) {
	draw_frame.active_z_layer = zlayer
}
@(deferred_out=set_z_layer)
push_z_layer :: proc(zlayer: ZLayer) -> ZLayer {
	og := draw_frame.active_z_layer
	draw_frame.active_z_layer = zlayer
	return og
}

// below is the lower level draw rect stuff

draw_rect_projected :: proc(
	world_to_clip: Matrix4,
	size: Vector2,
	col: Vector4=COLOR_WHITE,
	uv: Vector4=DEFAULT_UV,
	img_id: Image_Id=.nil,
	color_override:= v4{0,0,0,0},
	z_layer := ZLayer.nil,
) {

	bl := v2{ 0, 0 }
	tl := v2{ 0, size.y }
	tr := v2{ size.x, size.y }
	br := v2{ size.x, 0 }

	uv0 := uv
	if uv == DEFAULT_UV {
		uv0 = images[img_id].atlas_uvs
	}

    if draw_frame.flip_v {
        uv0.y, uv0.w = uv0.w, uv0.y
    }

	tex_index :u8= images[img_id].tex_index
	if img_id == .nil {
		tex_index = 255
	}

	draw_quad_projected(world_to_clip, {bl, tl, tr, br}, {col, col, col, col}, {uv0.xy, uv0.xw, uv0.zw, uv0.zy}, {tex_index,tex_index,tex_index,tex_index}, {color_override,color_override,color_override,color_override}, z_layer = z_layer)

}

draw_quad_projected :: proc(
	world_to_clip:   Matrix4,
	positions:       [4]Vector2,
	colors:          [4]Vector4,
	uvs:             [4]Vector2,
	tex_indicies:       [4]u8,
	//flags:           [4]Quad_Flags,
	color_overrides: [4]Vector4,
	z_layer: ZLayer=.nil,
	//hsv:             [4]Vector3
) {
	using linalg

	if draw_frame.quad_count >= MAX_QUADS {
		log_error("max quads reached")
		return
	}

	z_layer0 := z_layer
	if z_layer0 == .nil {
		z_layer0 = draw_frame.active_z_layer
	}

	verts := cast(^[4]Vertex)&draw_frame.quads[draw_frame.quad_count];
	draw_frame.quad_count += 1;

	verts[0].pos = (world_to_clip * Vector4{positions[0].x, positions[0].y, 0.0, 1.0}).xy
	verts[1].pos = (world_to_clip * Vector4{positions[1].x, positions[1].y, 0.0, 1.0}).xy
	verts[2].pos = (world_to_clip * Vector4{positions[2].x, positions[2].y, 0.0, 1.0}).xy
	verts[3].pos = (world_to_clip * Vector4{positions[3].x, positions[3].y, 0.0, 1.0}).xy

	verts[0].col = colors[0]
	verts[1].col = colors[1]
	verts[2].col = colors[2]
	verts[3].col = colors[3]

	verts[0].uv = uvs[0]
	verts[1].uv = uvs[1]
	verts[2].uv = uvs[2]
	verts[3].uv = uvs[3]

	verts[0].tex_index = tex_indicies[0]
	verts[1].tex_index = tex_indicies[1]
	verts[2].tex_index = tex_indicies[2]
	verts[3].tex_index = tex_indicies[3]

	verts[0].color_override = color_overrides[0]
	verts[1].color_override = color_overrides[1]
	verts[2].color_override = color_overrides[2]
	verts[3].color_override = color_overrides[3]

	verts[0].z_layer = u8(z_layer0)
	verts[1].z_layer = u8(z_layer0)
	verts[2].z_layer = u8(z_layer0)
	verts[3].z_layer = u8(z_layer0)
}

//
// :IMAGE STUFF
//
Image_Id :: enum {
	nil,
	player,
	bow,
	background,
	midground,
	foreground,
	dummy_hit1,
	dummy_hit2,
	dummy_hit3,
	dummy_hit4,
	dummy_hit5,
	dummy_1,
	dummy1_hit1,
	dummy1_hit2,
	dummy1_hit3,
	dummy1_hit4,
	dummy1_hit5,
	dummy_2,
	dummy2_hit1,
	dummy2_hit2,
	dummy2_hit3,
	dummy2_hit4,
	dummy2_hit5,
	dummy_3,
	dummy3_hit1,
	dummy3_hit2,
	dummy3_hit3,
	dummy3_hit4,
	dummy3_hit5,
	arrow,
	elemental_arrow,
	skills_button,
	skills_panel_bg,
	skills_panel_bg1,
	tooltip_bg,
    next_skill_panel_bg,
    next_skill_button_bg,
    next_skill_button_active_bg,
    radio_selected,
    radio_unselected,
    quests_button,
    coin,
    skill_xp_bar,
    back_focus,
    front_focus,
    bar,
    upgrades_button
}

Image :: struct {
	width, height: i32,
	tex_index: u8,
	sg_img: sg.Image,
	data: [^]byte,
	atlas_uvs: Vector4,
}
images: [128]Image
image_count: int

init_images :: proc() {
	using fmt

	img_dir := "res/images/"

	highest_id := 0;
	for img_name, id in Image_Id {
		if id == 0 { continue }

		if id > highest_id {
			highest_id = id
		}

		path := tprint(img_dir, img_name, ".png", sep="")
		png_data, succ := os.read_entire_file(path)
		assert(succ, tprint(path, "not found"))

		stbi.set_flip_vertically_on_load(1)
		width, height, channels: i32
		img_data := stbi.load_from_memory(raw_data(png_data), auto_cast len(png_data), &width, &height, &channels, 4)
		assert(img_data != nil, "stbi load failed, invalid image?")

		img : Image;
		img.width = width
		img.height = height
		img.data = img_data

		images[id] = img
	}
	image_count = highest_id + 1

	pack_images_into_atlas()
}

Atlas :: struct {
	w, h: int,
	sg_image: sg.Image,
}
atlas: Atlas

pack_images_into_atlas :: proc() {
    max_width := 0
    max_height := 0
    total_area := 0

    for img, id in images {
        if img.width == 0 do continue
        max_width = max(max_width, int(img.width))
        max_height = max(max_height, int(img.height))
        total_area += int(img.width) * int(img.height)
    }

    min_size := 128
    for min_size * min_size < total_area * 2 {
        min_size *= 2
    }

    atlas.w = min_size
    atlas.h = min_size

    nodes := make([dynamic]stbrp.Node, atlas.w)
    defer delete(nodes)

    cont: stbrp.Context
    stbrp.init_target(&cont, auto_cast atlas.w, auto_cast atlas.h, raw_data(nodes), auto_cast len(nodes))

    rects := make([dynamic]stbrp.Rect)
    defer delete(rects)

    for img, id in images {
        if img.width == 0 do continue
        rect := stbrp.Rect{
            id = auto_cast id,
            w = auto_cast img.width,
            h = auto_cast img.height,
        }
        append(&rects, rect)
    }

    if len(rects) == 0 {
        return
    }

    succ := stbrp.pack_rects(&cont, raw_data(rects), auto_cast len(rects))
    if succ == 0 {
        for rect, i in rects {
            fmt.printf("Rect %d: %dx%d = %d pixels\n",
                rect.id, rect.w, rect.h, rect.w * rect.h)
        }
        assert(false, "failed to pack all the rects, ran out of space?")
    }

    // allocate big atlas with proper size
    raw_data_size := atlas.w * atlas.h * 4
    atlas_data, err := mem.alloc(raw_data_size)
    if err != nil {
        return
    }
    defer mem.free(atlas_data)

    mem.set(atlas_data, 255, raw_data_size)

    // copy rect row-by-row into destination atlas
    for rect in rects {
        img := &images[rect.id]
        if img == nil || img.data == nil {
            continue
        }

        // copy row by row into atlas
        for row in 0 ..< rect.h {
            src_row := mem.ptr_offset(&img.data[0], row * rect.w * 4)
            dest_row := mem.ptr_offset(
                cast(^u8)atlas_data,
                ((rect.y + row) * auto_cast atlas.w + rect.x) * 4,
            )
            mem.copy(dest_row, src_row, auto_cast rect.w * 4)
        }

        stbi.image_free(img.data)
        img.data = nil

        img.atlas_uvs.x = cast(f32)rect.x / cast(f32)atlas.w
        img.atlas_uvs.y = cast(f32)rect.y / cast(f32)atlas.h
        img.atlas_uvs.z = img.atlas_uvs.x + cast(f32)img.width / cast(f32)atlas.w
        img.atlas_uvs.w = img.atlas_uvs.y + cast(f32)img.height / cast(f32)atlas.h
    }

    // Write debug atlas
    stbi.write_png(
        "./atlases/atlas.png",
        auto_cast atlas.w,
        auto_cast atlas.h,
        4,
        atlas_data,
        4 * auto_cast atlas.w,
    )

    // setup image for GPU
    desc: sg.Image_Desc
    desc.width = auto_cast atlas.w
    desc.height = auto_cast atlas.h
    desc.pixel_format = .RGBA8
    desc.data.subimage[0][0] = {
        ptr = atlas_data,
        size = auto_cast raw_data_size,
    }

    atlas.sg_image = sg.make_image(desc)
    if atlas.sg_image.id == sg.INVALID_ID {
        log_error("failed to make image")
    }
}

//
// :FONT
//
draw_text :: proc(pos: Vector2, text: string, col:=COLOR_WHITE, scale:= 1.0, pivot:=Pivot.bottom_left, z_layer:= ZLayer.nil) {
	using stbtt

	push_z_layer(z_layer)

	total_size : v2
	for char, i in text {

		advance_x: f32
		advance_y: f32
		q: aligned_quad
		GetBakedQuad(&font.char_data[0], font_bitmap_w, font_bitmap_h, cast(i32)char - 32, &advance_x, &advance_y, &q, false)

		size := v2{ abs(q.x0 - q.x1), abs(q.y0 - q.y1) }

		bottom_left := v2{ q.x0, -q.y1 }
		top_right := v2{ q.x1, -q.y0 }
		assert(bottom_left + size == top_right)

		if i == len(text)-1 {
			total_size.x += size.x
		} else {
			total_size.x += advance_x
		}

		total_size.y = max(total_size.y, top_right.y)
	}

	pivot_offset := total_size * -scale_from_pivot(pivot)

	debug_text := false
	if debug_text {
		draw_rect_aabb(pos + pivot_offset, total_size, col=COLOR_BLACK)
	}

	x: f32
	y: f32
	for char in text {

		advance_x: f32
		advance_y: f32
		q: aligned_quad
		GetBakedQuad(&font.char_data[0], font_bitmap_w, font_bitmap_h, cast(i32)char - 32, &advance_x, &advance_y, &q, false)

		size := v2{ abs(q.x0 - q.x1), abs(q.y0 - q.y1) }

		bottom_left := v2{ q.x0, -q.y1 }
		top_right := v2{ q.x1, -q.y0 }
		assert(bottom_left + size == top_right)

		offset_to_render_at := v2{x,y} + bottom_left

		offset_to_render_at += pivot_offset

		uv := v4{ q.s0, q.t1,
							q.s1, q.t0 }

		xform := Matrix4(1)
		xform *= xform_translate(pos)
		xform *= xform_scale(v2{auto_cast scale, auto_cast scale})
		xform *= xform_translate(offset_to_render_at)

		if debug_text {
			draw_rect_xform(xform, size, col=v4{1,1,1,0.8})
		}

		draw_rect_xform(xform, size, uv=uv, img_id=font.img_id, col=col)

		x += advance_x
		y += -advance_y
	}

}

font_bitmap_w :: 256
font_bitmap_h :: 256
char_count :: 96
Font :: struct {
	char_data: [char_count]stbtt.bakedchar,
	img_id: Image_Id,
}
font: Font

init_fonts :: proc() {
	using stbtt

	bitmap, _ := mem.alloc(font_bitmap_w * font_bitmap_h)
	font_height := 15
	path := "res/fonts/alagard.ttf"
	ttf_data, err := os.read_entire_file(path)
	assert(ttf_data != nil, "failed to read font")

	ret := BakeFontBitmap(raw_data(ttf_data), 0, auto_cast font_height, auto_cast bitmap, font_bitmap_w, font_bitmap_h, 32, char_count, &font.char_data[0])
	assert(ret > 0, "not enough space in bitmap")

	stbi.write_png("font.png", auto_cast font_bitmap_w, auto_cast font_bitmap_h, 1, bitmap, auto_cast font_bitmap_w)

	desc : sg.Image_Desc
	desc.width = auto_cast font_bitmap_w
	desc.height = auto_cast font_bitmap_h
	desc.pixel_format = .R8
	desc.data.subimage[0][0] = {ptr=bitmap, size=auto_cast (font_bitmap_w*font_bitmap_h)}
	sg_img := sg.make_image(desc)
	if sg_img.id == sg.INVALID_ID {
		log_error("failed to make image")
	}

	id := store_image(font_bitmap_w, font_bitmap_h, 1, sg_img)
	font.img_id = id
}

store_image :: proc(w: int, h: int, tex_index: u8, sg_img: sg.Image) -> Image_Id {

	img : Image
	img.width = auto_cast w
	img.height = auto_cast h
	img.tex_index = tex_index
	img.sg_img = sg_img
	img.atlas_uvs = DEFAULT_UV

	id := image_count
	images[id] = img
	image_count += 1

	return auto_cast id
}

//
// :game state
//

Game_State :: struct {
	ticks: u64,
	entities: [128]Entity,
	latest_entity_id: u64,
	player_handle: Entity_Handle,
	ui_config: UI_Config,
	ui_hot_reload: UI_Hot_Reload,
	skills_system: Skills_System,
    quests_system: Quests_System,
    upgrades_system: Upgrades_System,
    ui: struct {
        skills_button_scale: f32,
        upgrades_button_scale: f32,
        quest_button_scale: f32,
        skills_menu_alpha: f32,
        skills_tooltip_alpha: f32,
        skills_hover_tooltip_active: bool,
        skills_hover_tooltip_skill: ^Skill,
        skills_menu_pos: Vector2,
        last_mouse_pos: Vector2,
        skills_scroll_pos: f32,
        skills_scroll_initialized: bool,
    },
    upgrades_menu_open: bool,
}
gs: ^Game_State

// :update

get_player :: proc() -> ^Entity {
	return handle_to_entity(gs.player_handle)
}

get_delta_time :: proc() -> f64 {
	return sapp.frame_duration()
}

game_res_w :: 1280
game_res_h :: 720

update :: proc() {
	using linalg

	dt := sapp.frame_duration()

    focus_mode_skill_update(f32(dt))

	if gs.ticks == 0 {
	    // CREATE PLAYER
		en := entity_create()
		setup_player(en)
		gs.player_handle = entity_to_handle(en^)
	}

    width := sapp.width()
    height := sapp.height()
    update_projection(int(width), int(height))

	for &en in gs.entities {
		en.frame = {}
	}

	player := get_player()

	update_passive_skill_xp(f32(dt))

    // UPDATE ENTITIES
	for &en in gs.entities {
        if .allocated in en.flags {
            #partial switch en.kind {
                case .player: update_player(&en, f32(dt))
                case .arrow: update_arrow(&en, f32(dt))
                case .dummy: {
                    update_current_animation(&en.animations, f32(dt))
                    if .ghost_dummy in en.flags {
                        en.ghost_timer -= f32(dt)
                        if en.ghost_timer <= 0 {
                            entity_destroy(&en, f32(dt))
                        }
                    }
                }
            }
        }
	}

    check_and_reload(&gs.ui_hot_reload)
    gs.ui_config = gs.ui_hot_reload.config

	check_spawn_button()
	update_quests_system(&gs.quests_system, f32(dt))
    update_floating_texts(f32(dt))
	update_ui_state(gs, f32(dt))

	gs.ticks += 1
}

render :: proc() {
	using linalg

    draw_rect_aabb(v2{ game_res_w * -0.5, game_res_h * -0.5}, v2{game_res_w, game_res_h}, img_id=.background, z_layer = .background)
    draw_rect_aabb(v2{ game_res_w * -0.5, game_res_h * -0.5}, v2{game_res_w, game_res_h}, img_id=.midground, z_layer = .midground)
    draw_rect_aabb(v2{ game_res_w * -0.5, game_res_h * -0.5}, v2{game_res_w, game_res_h}, img_id=.foreground, z_layer = .foreground)

    if !has_active_dummy() {
        cfg := gs.ui_config.spawner

        button_pos := v2{cfg.pos_x, cfg.pos_y}
        button_size := v2{cfg.size_x, cfg.size_y}
        button_bounds := v2{cfg.bounds_x, cfg.bounds_y}
        hover := is_point_in_rect(mouse_pos_in_world_space(), button_pos, button_bounds)

        draw_sprite_with_size(
            button_pos,
            button_size,
            img_id = !hover ? Image_Id.next_skill_button_bg : Image_Id.next_skill_button_active_bg,
            pivot = .center_center,
            z_layer = .ui,
        )

        text_pos := v2{cfg.txt_pos_x, cfg.txt_pos_y}
        draw_text(text_pos, "Spawn Dummy", scale = 1.0, z_layer = .ui)
    }

	for &en in gs.entities {
		if .allocated in en.flags {
			#partial switch en.kind {
				case .player: {
				    draw_player_at_pos(&en, v2{-435, -315})
				}
				case .dummy: {
				    draw_dummy_at_pos(&en)
				}
				case .arrow: {
				    draw_arrow_at_pos(&en)
				}
			}
		}
	}

    render_floating_texts()

    if gs.skills_system.is_unlocked {
        focus_mode_skill_render()
        render_skill_menu_button()

        has_upgrades_unlocked := false
        for &skill in gs.skills_system.skills {
            if skill.is_unlocked && skill.type == .strength_boost {
                if skill.level >= 2 {
                    has_upgrades_unlocked = true
                    render_upgrades_menu_button()
                }
                break
            }
        }

        draw_sprite(
            pos = v2{0,310},
            img_id = .bar,
            pivot = .center_center,
            z_layer = .ui,
        )

        if has_unlocked_quests(&gs.quests_system) {
            render_quest_menu_button()
        }

        if gs.skills_system.menu_open {
            render_skills_ui()
        } else if gs.quests_system.menu_open && has_unlocked_quests(&gs.quests_system) {
            render_quests_ui()
        } else if gs.upgrades_menu_open && has_upgrades_unlocked {
            render_upgrades_menu()
        }

        cfg := gs.ui_config.gold_display
        pos := v2{cfg.pos_x, cfg.pos_y}

        sprite_pos := v2{cfg.sprite_pos_x, cfg.sprite_pos_y}
        sprite_size := v2{cfg.sprite_size_x, cfg.sprite_size_y}

        draw_nores_sprite_with_size(
            sprite_pos,
            sprite_size,
            cfg.sprite,
            pivot = .center_center,
            z_layer = .ui,
        )

        text_pos := pos + v2{cfg.text_offset_x, cfg.text_offset_y}
        draw_text(
            text_pos,
            fmt.tprintf("%d Gold", gs.skills_system.gold),
            col = v4{0,0,0,1},
            scale = auto_cast cfg.text_scale,
            z_layer = .ui,
        )
    }

	gs.ticks += 1
}

focus_mode_skill_update :: proc(dt: f32) {
    if gs.skills_system.focus_mode_active {
        base_duration := FOCUS_MODE_DURATION

        extra_duration := f32(0)
        if gs.quests_system.active_quest != nil && gs.quests_system.active_quest.type == .meditation_training {
            extra_duration = f32(gs.quests_system.active_quest.level)
        }

        total_duration := f32(base_duration) + extra_duration

        gs.skills_system.focus_mode_timer -= f32(dt)
        if gs.skills_system.focus_mode_timer <= 0 {
            gs.skills_system.focus_mode_active = false

            if extra_duration > 0 {
                add_floating_text_params(
                    v2{-550, 250},
                    fmt.tprintf("Innder Focus ended (+%.0fs duration)", extra_duration),
                    v4{0.5, 0.3, 0.8, 1.0},
                    scale = 0.8,
                    target_scale = 1.0,
                    lifetime = 1.2,
                    velocity = v2{0, 75},
                )
            }
        }
    }

    if gs.skills_system.focus_mode_button_visible {
        gs.skills_system.focus_mode_button_timer -= f32(dt)
        if gs.skills_system.focus_mode_button_timer <= 0 {
            gs.skills_system.focus_mode_button_visible = false
        }
    }
}

focus_mode_skill_render :: proc() {
    if gs.skills_system.focus_mode_button_visible {
        cfg := gs.ui_config.skills
        button_pos := v2{cfg.focus_mode.button_pos_x, cfg.focus_mode.button_pos_y}
        button_size := v2{cfg.focus_mode.button_size_x, cfg.focus_mode.button_size_y}
        button_bounds := v2{cfg.focus_mode.button_bounds_x, cfg.focus_mode.button_bounds_y}
        alpha := min(1.0, gs.skills_system.focus_mode_button_timer)

        text_pos := button_pos + v2{cfg.focus_mode.text_pos_x, cfg.focus_mode.text_pos_y}

        glow_size := button_size * 1.2

        draw_sprite_with_size(
            button_pos,
            button_size,
            cfg.focus_mode.back_sprite,
            color_override = v4{0.5, 0.3, 0.8, 0.3 * alpha},
            pivot = .center_center,
            z_layer = .ui,
        )

        draw_sprite_with_size(
            button_pos,
            button_size,
            cfg.focus_mode.front_sprite,
            color_override = v4{0.4, 0.2, 0.6, alpha},
            pivot = .center_center,
            z_layer = .ui,
        )

        draw_text(
            text_pos,
            "Focus Mode!",
            col = v4{1, 1, 1, alpha},
            scale = auto_cast cfg.focus_mode.text_scale,
            pivot = .center_center,
            z_layer = .ui,
        )

        mouse_pos := mouse_pos_in_world_space()
        if is_point_in_rect(mouse_pos, button_pos, button_bounds) && key_just_pressed(.LEFT_MOUSE) {
            gs.skills_system.focus_mode_active = true

            base_duration := FOCUS_MODE_DURATION
            extra_duration := f32(0)

            if gs.quests_system.active_quest != nil && gs.quests_system.active_quest.type == .meditation_training {
                extra_duration = f32(gs.quests_system.active_quest.level)
            }

            gs.skills_system.focus_mode_timer = f32(base_duration) + extra_duration
            gs.skills_system.focus_mode_button_visible = false

            add_floating_text_params(
                button_pos + v2{0, 30},
                "Focus Mode Activated!",
                v4{0.5, 0.3, 0.8, 1.0},
                scale = 0.8,
                target_scale = 1.0,
                lifetime = 1.0,
                velocity = v2{0, 75},
            )
        }
    }

    if gs.skills_system.focus_mode_active {
        indicator_pos := v2{-550, 200}
        draw_text(
            indicator_pos,
            fmt.tprintf("Focus Mode: %.1fs", gs.skills_system.focus_mode_timer),
            col = v4{0.5, 0.3, 0.8, 1.0},
            scale = 1.2,
            z_layer = .ui,
        )
    }
}

draw_player_at_pos :: proc(en: ^Entity, pos: Vector2) {
    xform := Matrix4(1)
    xform *= xform_scale(v2{0.38, 0.38})
    draw_sprite(pos, .player, pivot = .bottom_center, xform=xform, z_layer=.player)

    bow_offset := v2{2, 25}
    bow_pos := pos + bow_offset

    bow_xform := Matrix4(1)
    bow_xform *= xform_translate(bow_pos)
    bow_xform *= xform_rotate(en.bow_angle)
    bow_xform *= xform_scale(v2{0.2, 0.2})

    draw_sprite(v2{0,0}, .bow, pivot = .center_left, xform = bow_xform, z_layer = .bow)
}

draw_dummy_at_pos :: proc(en: ^Entity){
    xform := Matrix4(1)
    xform *= xform_scale(v2{0.35, 0.35})

    color_override := .ghost_dummy in en.flags ? v4{1, 1, 1, 0.5} : v4{0, 0, 0, 0}

    health_percent := en.health / en.max_health

    current_tier := gs.upgrades_system.dummy_tiers[gs.upgrades_system.current_tier]

    base_pos := v2{en.pos.x, en.pos.y - 8}
    if en.animations.current_animation == "" {
        draw_sprite(
            base_pos,
            current_tier.base_sprite,
            pivot = .bottom_center,
            xform = xform,
            z_layer = .player,
            color_override = color_override,
        )
    } else {
        draw_current_animation(
            &en.animations,
            base_pos,
            pivot = .bottom_center,
            xform = xform,
            z_layer = .player,
            color_override = color_override,
        )
    }

    if en.health < en.max_health {
        bar_width := 64.0
        bar_height := 8.0
        health_ratio := en.health / en.max_health

        bar_pos := en.pos + v2{auto_cast -bar_width / 2, 100}
        draw_rect_aabb(bar_pos, v2{auto_cast bar_width, auto_cast bar_height}, col=v4{0.243,0.243,0.259,1}, z_layer = .ui)

        draw_rect_aabb(bar_pos, v2{auto_cast bar_width * health_ratio, auto_cast bar_height * 0.5}, col=v4{0,1,0,1}, z_layer = .ui)
    }

     if en.has_weak_point {
        is_tactical := gs.quests_system.active_quest != nil &&
                      gs.quests_system.active_quest.type == .tactical_assessment &&
                      gs.quests_system.active_quest.tactical_mode_active

        weak_point_world_pos := en.pos + en.weak_point_pos
        base_glow_size := en.weak_point_radius * 4
        glow_size := is_tactical ? base_glow_size * 1.5 : base_glow_size

        glow_color := is_tactical ? v4{1, 0.4, 0.4, en.weak_point_glow_intensity * 1.5} : v4{1, 0.6, 0, en.weak_point_glow_intensity}

        draw_rect_aabb(
            weak_point_world_pos - v2{glow_size, glow_size} * 0.5,
            v2{glow_size, glow_size},
            col = glow_color,
            z_layer = .player
        )

        center_size := v2{en.weak_point_radius * 2, en.weak_point_radius * 2}
        center_color := is_tactical ? v4{1, 0.6, 0.6, en.weak_point_glow_intensity + 0.4} : v4{1, 1.0, 0, en.weak_point_glow_intensity + 0.2}

        draw_rect_aabb(
            weak_point_world_pos - center_size * 0.5,
            center_size,
            col = center_color,
            z_layer = .player
        )
    }
}

draw_arrow_at_pos :: proc(en: ^Entity){
    xform := Matrix4(1)
    xform *= xform_translate(en.pos)
    xform *= xform_rotate(en.rotation)
    xform *= xform_scale(v2{0.12, 0.12})

    if en.arrow_data.arrow_type == .elemental{
        draw_sprite(v2{0,0}, .elemental_arrow, pivot = .center_center, xform = xform, z_layer = .player)
    }else{
        draw_sprite(v2{0,0}, .arrow, pivot = .center_center, xform = xform, z_layer = .player)
    }
}

mouse_pos_in_screen_space :: proc() -> Vector2 {
	if draw_frame.coord_space.proj == {} {
		log_error("no projection matrix set yet")
	}

	mouse := v2{app_state.input_state.mouse_x, app_state.input_state.mouse_y}
	x := mouse.x / f32(window_w);
	y := mouse.y / f32(window_h) - 1.0;
	y *= -1
	return v2{x * game_res_w, y * game_res_h}
}

mouse_pos_in_world_space :: proc() -> Vector2 {
    if draw_frame.coord_space.proj == {} {
        log_error("no projection matrix set yet")
        return v2{0, 0}
    }

    mouse := v2{app_state.input_state.mouse_x, app_state.input_state.mouse_y}

    width := f32(sapp.width())
    height := f32(sapp.height())

    ndc_x := (mouse.x / width) * 2.0 - 1.0
    ndc_y := -((mouse.y / height) * 2.0 - 1.0)

    mouse_clip := v4{ndc_x, ndc_y, 0, 1}

    view_proj := draw_frame.coord_space.proj * draw_frame.coord_space.camera
    inv_view_proj := linalg.inverse(view_proj)
    world_pos := mouse_clip * inv_view_proj

    return world_pos.xy
}

//
// :dummies
DUMMY_MAX_HEALTH :: 100.0
DUMMY_POSITIONS := []Vector2{
    {300, -320},
    {450, -320},
    {600, -320},
    {750, -320},
}

spawn_dummy :: proc(position: Vector2, is_ghost := false) -> ^Entity {
    dummy := entity_create()
    if dummy == nil do return nil

    setup_dummy(dummy, is_ghost)
    dummy.pos = position

    return dummy
}

//
// :player
update_player :: proc(player: ^Entity, dt: f32) {
    player.shoot_cooldown -= dt
    target := find_random_target()

    if target != nil {
        shoot_pos := v2{-440, -320} + v2{30, 40}

        target_pos := target.pos
        if target.has_weak_point {
            target_pos += target.weak_point_pos
        }

        dir := target_pos - shoot_pos
        angle := math.atan2(dir.y, dir.x)
        target_degrees := math.to_degrees(angle)

        for target_degrees < 0 do target_degrees += 360
        for target_degrees >= 360 do target_degrees -= 360

        player.target_bow_angle = target_degrees
    }

    rot_speed := 720.0 * dt
    angle_diff := player.target_bow_angle - player.bow_angle

    for angle_diff > 180 do angle_diff -= 360
    for angle_diff < -180 do angle_diff += 360

    if abs(angle_diff) < rot_speed {
        player.bow_angle = player.target_bow_angle
    } else {
        player.bow_angle += sign(angle_diff) * rot_speed
    }

    for player.bow_angle < 0 do player.bow_angle += 360
    for player.bow_angle >= 360 do player.bow_angle -= 360

    if target != nil && player.shoot_cooldown <= 0 {
        shoot_arrow(player, target)

        speed_multiplier := f32(1.0)
        for &skill in gs.skills_system.skills {
            if skill.is_unlocked && skill.type == .speed_boost {
                speed_bonus := calculate_skill_bonus(&skill)
                speed_multiplier = 1.0 / (1.0 + speed_bonus)
                break
            }
        }
        player.shoot_cooldown = SHOOT_COOLDOWN * speed_multiplier
    }
}

find_random_target :: proc() -> ^Entity {
    oldest_dummy: ^Entity
    oldest_time := f64(max(f64))

    for &en in gs.entities {
        if .allocated in en.flags && en.kind == .dummy {
            if en.spawn_time < f32(oldest_time) {
                oldest_time = f64(en.spawn_time)
                oldest_dummy = &en
            }
        }
    }

    return oldest_dummy
}

//
// :arrows

Arrow_Type :: enum {
    normal,
    elemental,
}

Arrow_Data :: struct {
    velocity: Vector2,
    target_pos: Vector2,
    lifetime: f32,
    arrow_type: Arrow_Type,
}

ARROW_BASE_DAMAGE :: 150.0 // 20
ARROW_DAMAGE :: 150.0 // 20
ELEMENTAL_ARROW_DAMAGE :: 40.0
SHOOT_COOLDOWN :: 1.2
ARROW_SPEED :: 1700.0
ARROW_LIFETIME :: 0.6
ACCURACY_VARIANCE :: 0.0
GRAVITY_EFFECT :: 6000.0
HIT_CHANCE :: 1.1
DUMMY_TARGET_WIDTH :: 64.0
DUMMY_TARGET_HEIGHT :: 64.0

update_arrow :: proc(e: ^Entity, dt: f32) {
    e.arrow_data.velocity.y -= GRAVITY_EFFECT * dt
    old_pos := e.pos
    e.pos += e.arrow_data.velocity * dt

    e.arrow_data.lifetime -= dt
    if e.arrow_data.lifetime <= 0 {
        entity_destroy(e, f32(dt))
        return
    }

    arrow_radius := f32(10.0)

    for &target in gs.entities {
        if .allocated not_in target.flags || target.kind != .dummy {
            continue
        }

        weak_point_hit := false
        if target.has_weak_point {
            weak_point_pos := target.pos + target.weak_point_pos
            if circle_collide(e.pos, weak_point_pos, arrow_radius, target.weak_point_radius * 4) {
                weak_point_hit = true
            }
        }

        dummy_radius := f32(32.0)
        dummy_hit := false
        if !weak_point_hit {
            dummy_center := target.pos + v2{0, 32}
            if circle_collide(e.pos, dummy_center, arrow_radius, dummy_radius) {
                dummy_hit = true
            }
        }

        if weak_point_hit || dummy_hit {
            is_elemental := e.arrow_data.arrow_type == .elemental
            damage: f32 = 0.0

            if is_elemental && weak_point_hit {
                damage = ELEMENTAL_ARROW_DAMAGE * 2
            }else if !is_elemental && weak_point_hit {
                damage = ARROW_DAMAGE * 2
            }else if is_elemental && !weak_point_hit {
                damage = ELEMENTAL_ARROW_DAMAGE
            }else {
                damage = ARROW_DAMAGE
            }

            if weak_point_hit {
                add_floating_text_params(
                    target.pos + v2{0, 50},
                    "WEAK POINT!",
                    v4{1, 0.5, 0, 1},
                    scale = 1.2,
                    target_scale = 1.5,
                    lifetime = 1.0,
                    velocity = v2{0, 100},
                )
            }

            damage_entity(&target, damage, is_elemental)
            entity_destroy(e, f32(dt))
            return
        }
    }

    angle := math.atan2(e.arrow_data.velocity.y, e.arrow_data.velocity.x)
    e.rotation = math.to_degrees(angle)
}

shoot_arrow :: proc(player: ^Entity, target: ^Entity){
    multi_shot_chance := f32(0)
    mystic_shot_chance := f32(0)
    for &skill in gs.skills_system.skills {
        if skill.is_unlocked {
            #partial switch skill.type {
                case .storm_arrows:
                    multi_shot_chance = calculate_skill_bonus(&skill)
                case .mystic_fletcher:
                    mystic_shot_chance = calculate_skill_bonus(&skill)
            }
        }
    }

    spawn_single_arrow(player, target, .normal)

    if rand.float32() < multi_shot_chance {
        spawn_single_arrow(player, target, .normal)
    }

    if rand.float32() < mystic_shot_chance {
        spawn_single_arrow(player, target, .elemental)
    }
}

spawn_single_arrow :: proc(player: ^Entity, target: ^Entity, arrow_type: Arrow_Type) {
    arrow := entity_create()
    if arrow == nil do return

    shoot_pos := v2{-440, -320} + v2{10, 25}

    target_pos: Vector2
    if target.has_weak_point {
        weak_point_pos := target.pos + target.weak_point_pos
        variation := v2{
            rand.float32_range(-target.weak_point_radius, target.weak_point_radius),
            rand.float32_range(-target.weak_point_radius, target.weak_point_radius),
        }
        target_pos = weak_point_pos + variation
    } else {
        dummy_center := target.pos + v2{0, 25}
        target_variation := v2{
            rand.float32_range(-DUMMY_TARGET_WIDTH/2, DUMMY_TARGET_WIDTH/2),
            rand.float32_range(-DUMMY_TARGET_HEIGHT/2, DUMMY_TARGET_HEIGHT/2),
        }

        accurate_shot := rand.float32() < HIT_CHANCE
        if accurate_shot {
            target_pos = dummy_center + target_variation
        } else {
            miss_variation := v2{
                rand.float32_range(-ACCURACY_VARIANCE, ACCURACY_VARIANCE),
                rand.float32_range(-ACCURACY_VARIANCE, ACCURACY_VARIANCE),
            }
            target_pos = dummy_center + target_variation + miss_variation
        }
    }

    direction := target_pos - shoot_pos
    distance := linalg.length(direction)
    flight_time := distance / ARROW_SPEED

    direction_normalized := direction / distance

    vertical_boost := GRAVITY_EFFECT * flight_time * 0.5

    base_velocity := direction_normalized * ARROW_SPEED
    initial_velocity := base_velocity + v2{0, vertical_boost}

    setup_arrow(arrow, shoot_pos, target_pos, arrow_type)
    arrow.arrow_data.velocity = initial_velocity
    player.arrow_count += 1
}

calculate_arrow_damage :: proc(system: ^Skills_System) -> f32 {
    base_damage := ARROW_BASE_DAMAGE
    final_damage := base_damage

    for &skill in system.skills {
        if skill.is_unlocked {
            #partial switch skill.type {
                case .strength_boost: {
                    strength_bonus := 0.10 + 0.10 * f32(skill.level - 1)
                    final_damage *= f64(1.0 + strength_bonus)
                }
            }
        }
    }

    return f32(final_damage)
}

//
// :entity
//

Entity_Flags :: enum {
	allocated,
	ghost_dummy,
	//physics
}

Entity_Kind :: enum {
	nil,
	player,
	dummy,
	arrow,
}

Entity :: struct {
	handle: Entity_Handle,
	kind: Entity_Kind,
	flags: bit_set[Entity_Flags],
	pos: Vector2,
    animations: Animation_Collection,
    rotation: f32,
	frame: struct{
		input_axis: Vector2,
	},
	arrow_data: Arrow_Data,
	shoot_cooldown: f32,
	arrow_count: int,
    max_arrows: int,
	health: f32,
	max_health: f32,
	aabb: Vector4,
	img_id: Image_Id,
	bow_angle: f32,
	target_bow_angle: f32,
	ghost_timer: f32,
    has_weak_point: bool,
    weak_point_pos: Vector2,
    weak_point_radius: f32,
    weak_point_glow_intensity: f32,
    spawn_time: f32,
}

entity_data: [Entity_Kind]Entity

Entity_Handle :: struct {
    id: u64,
    index: int,
}

damage_entity :: proc(e: ^Entity, base_damage: f32, is_elemental := false) {
    damage := base_damage
    crit_occurred := false

    if system := &gs.skills_system; system.is_unlocked {
        for &skill in system.skills {
            if skill.is_unlocked && skill.type == .strength_boost {
                strength_bonus := 0.10 + 0.10 * f32(skill.level - 1)
                damage *= f32(1.0 + strength_bonus)
            }
        }

        for &skill in system.skills {
            if skill.is_unlocked && skill.type == .critical_boost {
                crit_chance := calculate_skill_bonus(&skill)
                if rand.float32() < crit_chance {
                    crit_occurred = true
                }
                break
            }
        }

        if crit_occurred {
            damage *= 2.0

            for &skill in system.skills {
                if skill.is_unlocked && skill.type == .warrior_stamina {
                    instant_shot_chance := calculate_skill_bonus(&skill)
                    if rand.float32() < instant_shot_chance {
                        player := get_player()
                        target := find_random_target()
                        if player != nil && target != nil {
                            old_max := player.max_arrows
                            player.max_arrows += 1

                            shoot_arrow(player, target)
                            player.max_arrows = old_max

                            player.shoot_cooldown = 0

                            add_floating_text_params(
                                player.pos + v2{0, 50},
                                "Warrior's Stamina!",
                                v4{1,0.8,0,1},
                                scale = 0.8,
                                target_scale = 1.0,
                                lifetime = 1.0,
                                velocity = v2{0, 75},
                            )
                        }
                    }
                }
            }
        }
    }

    if e.kind == .dummy && e.has_weak_point {
        hit_pos := e.pos + e.weak_point_pos
        arrow_to_weak_point := hit_pos - e.pos
        if linalg.length(arrow_to_weak_point) <= e.weak_point_radius {
            if gs.quests_system.active_quest != nil &&
               gs.quests_system.active_quest.type == .tactical_assessment &&
               gs.quests_system.active_quest.tactical_mode_active {

                quest := gs.quests_system.active_quest
                combo_bonus := f32(quest.tactical_mode_combo) * 0.5
                bonus_gold := int(f32(damage) * (1.0 + combo_bonus))
                gs.skills_system.gold += bonus_gold

                quest.tactical_mode_combo += 1

                combo_text := fmt.tprintf("Tactical Hit x%d! +%d Gold",
                    quest.tactical_mode_combo, bonus_gold)
                add_floating_text_params(
                    e.pos + v2{0, 70},
                    combo_text,
                    v4{1, 0.3, 0.3, 1.0},
                    scale = 0.8,
                    target_scale = 1.0,
                    lifetime = 1.0,
                    velocity = v2{0, 75},
                )
            }
        }
    }

    if e.kind == .dummy {
        reset_and_play_animation(&e.animations, "hit")
    }

    text_pos := e.pos + v2{-40, 20}

    pos_offset_x := rand.float32_range(-20, 20)
    pos_offset_y := rand.float32_range(-20, 20)
    text_pos += v2{pos_offset_x, pos_offset_y}

    velocity_x := rand.float32_range(-100, 100)
    velocity_y := rand.float32_range(100, 200)
    text_color := v4{1,1,1,1}
    text_scale := f32(0.6)
    target_scale := f32(1.0)

    if is_elemental {
        text_color = v4{0.74, 0.09, 0.64, 1}
    } else if crit_occurred {
        text_color = v4{1,0,0,1}
        target_scale = 1.2
    }

    add_floating_text_params(
        text_pos,
        fmt.tprintf("%d", int(damage)),
        color = text_color,
        scale = text_scale,
        target_scale = target_scale,
        lifetime = 0.65,
        velocity = v2{velocity_x, velocity_y},
    )

    e.health -= damage
    if e.health <= 0 {
        entity_destroy(e)
    }
}

setup_entity :: proc(e: ^Entity, kind: Entity_Kind){
    #partial switch kind{
        case .nil: return
        case .player: setup_player(e)
        case .dummy: setup_dummy(e)
        case .arrow: {
            e.kind = .arrow
            e.flags |= {.allocated}
        }
    }
}

handle_to_entity :: proc(handle: Entity_Handle) -> ^Entity {
    en := &gs.entities[handle.index]
    if en.handle.id == handle.id{
        return en
    }
	log_error("entity no longer valid")
	return nil
}

entity_to_handle :: proc(entity: Entity) -> Entity_Handle {
	return entity.handle
}

entity_create :: proc() -> ^Entity {
	spare_en : ^Entity
	index := -1
	for &en, i in gs.entities {
		if !(.allocated in en.flags) {
			spare_en = &en
			index = i
			break
		}
	}

	if spare_en == nil {
		log_error("ran out of entities, increase size")
		return nil
	} else {
		spare_en.flags = { .allocated }
		gs.latest_entity_id += 1
		spare_en.handle.id = gs.latest_entity_id
		spare_en.handle.index = index
		return spare_en
	}
}

entity_destroy :: proc(entity: ^Entity, dt: f32 = 0) {
    if entity.kind == .dummy {
    gs.upgrades_system.frames_since_last_spawn = 0
        if .ghost_dummy not_in entity.flags {
            for &skill in gs.skills_system.advanced_skills {
                if skill.is_unlocked && skill.type == .formation_mastery {
                    ghost_chance := calculate_skill_bonus(&skill)
                    if rand.float32() < ghost_chance {
                        ghost_pos := entity.pos
                        ghost := spawn_dummy(ghost_pos, true)
                        if ghost != nil {
                            add_floating_text_params(
                                ghost_pos + v2{0, 50},
                                "Ghost Dummy Spawned!",
                                v4{0.7, 0.7, 1.0, 1.0},
                                scale = 0.8,
                                target_scale = 1.0,
                                lifetime = 1.0,
                                velocity = v2{0, 75},
                            )
                        }
                    }
                }

                if skill.is_unlocked && skill.type == .battle_meditation {
                    if !gs.skills_system.focus_mode_active && !gs.skills_system.focus_mode_button_visible {
                        trigger_chance := calculate_skill_bonus(&skill)
                        if rand.float32() < trigger_chance {
                            gs.skills_system.focus_mode_button_visible = true
                            gs.skills_system.focus_mode_button_timer = FOCUS_MODE_BUTTON_DURATION
                        }
                    }
                }
            }
        }

        gs.skills_system.dummies_killed += 1
        check_skills_unlock(&gs.skills_system)

        if gs.skills_system.is_unlocked {
            base_xp := 50
            if .ghost_dummy in entity.flags {
                base_xp *= 2
            }

            total_xp := add_xp_to_active_skill(&gs.skills_system, base_xp)

            pos := entity.pos + v2{0, 50}
            add_floating_text_params(
                pos,
                fmt.tprintf("+%d XP", total_xp),
                scale = 0.8,
                target_scale = 1.0,
                lifetime = 1.0,
                velocity = v2{0, 75},
            )
        }
    }

    mem.set(entity, 0, size_of(Entity))
}


//
// :setups

setup_player :: proc(e: ^Entity) {
    e.kind = .player
    e.flags |= { .allocated }
    e.max_arrows = 1
    e.arrow_count = 0
    e.bow_angle = 0
    e.target_bow_angle = 0
}

setup_dummy :: proc(e: ^Entity, is_ghost := false) {
    e.kind = .dummy
    e.flags |= { .allocated }
    e.spawn_time = f32(seconds_since_init())

    current_tier := gs.upgrades_system.dummy_tiers[gs.upgrades_system.current_tier]
    e.health = current_tier.health
    e.max_health = current_tier.health

    if is_ghost {
        e.flags |= { .ghost_dummy }
        e.ghost_timer = 5.0
    }

    animations := create_animation_collection()
    hit_anim := create_animation(current_tier.frames, 0.08, false, "hit")

    add_animation(&animations, hit_anim)
    e.animations = animations

    dummy_size := v2{64, 64}
    e.aabb = aabb_make(e.pos, dummy_size, Pivot.bottom_center)

    e.has_weak_point = false
    e.weak_point_radius = 6.0
    e.weak_point_glow_intensity = 0.0

    weak_point_chance := f32(0)
    for &skill in gs.skills_system.advanced_skills {
        if skill.is_unlocked && skill.type == .tactical_analysis {
            weak_point_chance = calculate_skill_bonus(&skill)
            break
        }
    }

    if rand.float32() < weak_point_chance {
        e.has_weak_point = true
        offset_x := rand.float32_range(-20, -5)
        offset_y := rand.float32_range(20, 60)
        e.weak_point_pos = v2{offset_x, offset_y}
    }
}

setup_arrow :: proc(e: ^Entity, start_pos: Vector2, target_pos: Vector2, arrow_type := Arrow_Type.normal) {
    e.kind = .arrow
    e.flags |= {.allocated}
    e.pos = start_pos

    direction := target_pos - start_pos
    distance := linalg.length(direction)
    flight_time := distance / ARROW_SPEED

    direction_normalized := direction / distance
    base_velocity := direction_normalized * ARROW_SPEED

    vertical_boost := GRAVITY_EFFECT * flight_time * 0.5

    e.arrow_data = Arrow_Data {
        velocity = base_velocity + v2{0, vertical_boost},
        target_pos = target_pos,
        lifetime = ARROW_LIFETIME,
        arrow_type = arrow_type,
    }
}

//
// :input

Key_Code :: enum {
	INVALID = 0,
	SPACE = 32,
	APOSTROPHE = 39,
	COMMA = 44,
	MINUS = 45,
	PERIOD = 46,
	SLASH = 47,
	_0 = 48,
	_1 = 49,
	_2 = 50,
	_3 = 51,
	_4 = 52,
	_5 = 53,
	_6 = 54,
	_7 = 55,
	_8 = 56,
	_9 = 57,
	SEMICOLON = 59,
	EQUAL = 61,
	A = 65,
	B = 66,
	C = 67,
	D = 68,
	E = 69,
	F = 70,
	G = 71,
	H = 72,
	I = 73,
	J = 74,
	K = 75,
	L = 76,
	M = 77,
	N = 78,
	O = 79,
	P = 80,
	Q = 81,
	R = 82,
	S = 83,
	T = 84,
	U = 85,
	V = 86,
	W = 87,
	X = 88,
	Y = 89,
	Z = 90,
	LEFT_BRACKET = 91,
	BACKSLASH = 92,
	RIGHT_BRACKET = 93,
	GRAVE_ACCENT = 96,
	WORLD_1 = 161,
	WORLD_2 = 162,
	ESCAPE = 256,
	ENTER = 257,
	TAB = 258,
	BACKSPACE = 259,
	INSERT = 260,
	DELETE = 261,
	RIGHT = 262,
	LEFT = 263,
	DOWN = 264,
	UP = 265,
	PAGE_UP = 266,
	PAGE_DOWN = 267,
	HOME = 268,
	END = 269,
	CAPS_LOCK = 280,
	SCROLL_LOCK = 281,
	NUM_LOCK = 282,
	PRINT_SCREEN = 283,
	PAUSE = 284,
	F1 = 290,
	F2 = 291,
	F3 = 292,
	F4 = 293,
	F5 = 294,
	F6 = 295,
	F7 = 296,
	F8 = 297,
	F9 = 298,
	F10 = 299,
	F11 = 300,
	F12 = 301,
	F13 = 302,
	F14 = 303,
	F15 = 304,
	F16 = 305,
	F17 = 306,
	F18 = 307,
	F19 = 308,
	F20 = 309,
	F21 = 310,
	F22 = 311,
	F23 = 312,
	F24 = 313,
	F25 = 314,
	KP_0 = 320,
	KP_1 = 321,
	KP_2 = 322,
	KP_3 = 323,
	KP_4 = 324,
	KP_5 = 325,
	KP_6 = 326,
	KP_7 = 327,
	KP_8 = 328,
	KP_9 = 329,
	KP_DECIMAL = 330,
	KP_DIVIDE = 331,
	KP_MULTIPLY = 332,
	KP_SUBTRACT = 333,
	KP_ADD = 334,
	KP_ENTER = 335,
	KP_EQUAL = 336,
	LEFT_SHIFT = 340,
	LEFT_CONTROL = 341,
	LEFT_ALT = 342,
	LEFT_SUPER = 343,
	RIGHT_SHIFT = 344,
	RIGHT_CONTROL = 345,
	RIGHT_ALT = 346,
	RIGHT_SUPER = 347,
	MENU = 348,

	LEFT_MOUSE = 400,
	RIGHT_MOUSE = 401,
	MIDDLE_MOUSE = 402,
}
MAX_KEYCODES :: sapp.MAX_KEYCODES
map_sokol_mouse_button :: proc "c" (sokol_mouse_button: sapp.Mousebutton) -> Key_Code {
	#partial switch sokol_mouse_button {
		case .LEFT: return .LEFT_MOUSE
		case .RIGHT: return .RIGHT_MOUSE
		case .MIDDLE: return .MIDDLE_MOUSE
	}
	return nil
}

Input_State_Flags :: enum {
	down,
	just_pressed,
	just_released,
	repeat,
}

Input_State :: struct {
	keys: [MAX_KEYCODES]bit_set[Input_State_Flags],
	mouse_x, mouse_y: f32,
}

reset_input_state_for_next_frame :: proc(state: ^Input_State) {
	for &set in state.keys {
		set -= {.just_pressed, .just_released, .repeat}
	}
}

key_just_pressed :: proc(code: Key_Code) -> bool {
	return .just_pressed in app_state.input_state.keys[code]
}
key_down :: proc(code: Key_Code) -> bool {
	return .down in app_state.input_state.keys[code]
}
key_just_released :: proc(code: Key_Code) -> bool {
	return .just_released in app_state.input_state.keys[code]
}
key_repeat :: proc(code: Key_Code) -> bool {
	return .repeat in app_state.input_state.keys[code]
}

event :: proc "c" (event: ^sapp.Event) {
    context = runtime.default_context()
	input_state := &app_state.input_state

	#partial switch event.type {
	    case .RESIZED:
	       update_projection(auto_cast event.window_width, auto_cast event.window_height)
		case .MOUSE_UP:
		if .down in input_state.keys[map_sokol_mouse_button(event.mouse_button)] {
			input_state.keys[map_sokol_mouse_button(event.mouse_button)] -= { .down }
			input_state.keys[map_sokol_mouse_button(event.mouse_button)] += { .just_released }
		}
		case .MOUSE_DOWN:
		if !(.down in input_state.keys[map_sokol_mouse_button(event.mouse_button)]) {
			input_state.keys[map_sokol_mouse_button(event.mouse_button)] += { .down, .just_pressed }
		}

		case .MOUSE_MOVE:
		input_state.mouse_x = event.mouse_x
		input_state.mouse_y = event.mouse_y

		case .KEY_UP:
		if .down in input_state.keys[event.key_code] {
			input_state.keys[event.key_code] -= { .down }
			input_state.keys[event.key_code] += { .just_released }
		}
		case .KEY_DOWN:
		if !event.key_repeat && !(.down in input_state.keys[event.key_code]) {
			input_state.keys[event.key_code] += { .down, .just_pressed }
		}
		if event.key_repeat {
			input_state.keys[event.key_code] += { .repeat }
		}
	}
}

update_projection :: proc(width, height: int) {
    using linalg
    draw_frame.coord_space.proj = matrix_ortho3d_f32(
        game_res_w * -0.5,
        game_res_w * 0.5,
        game_res_h * -0.5,
        game_res_h * 0.5,
        -1,
        1,
    )
    draw_frame.coord_space.camera = Matrix4(1)
}

//
// :animations
Animation_State :: enum {
    Playing,
    Paused,
    Stopped,
}

Animation :: struct {
    frames: []Image_Id,
    current_frame: int,
    frame_duration: f32,
    frame_timer: f32,
    state: Animation_State,
    loops: bool,
    name: string,
    base_duration: f32,
}

Animation_Collection :: struct {
    animations: map[string]Animation,
    current_animation: string,
}

create_animation :: proc(frames: []Image_Id, frame_duration: f32, loops: bool, name: string) -> Animation {
    frames_copy := make([]Image_Id, len(frames), context.allocator)
    copy(frames_copy[:], frames)

    return Animation{
        frames = frames_copy,
        current_frame = 0,
        frame_duration = frame_duration,
        base_duration = frame_duration,
        frame_timer = 0,
        state = .Stopped,
        loops = loops,
        name = name,
    }
}

adjust_animation_to_speed :: proc(anim:  ^Animation, speed_multiplier: f32) {
    if anim == nil do return

    anim.frame_duration = anim.base_duration / speed_multiplier
}

update_animation :: proc(anim: ^Animation, delta_t: f32) -> bool {
    if anim == nil {
        return false
    }

    if anim.state != .Playing {
        return false
    }

    anim.frame_timer += delta_t
    if anim.frame_timer >= anim.frame_duration {
        anim.frame_timer -= anim.frame_duration
        next_frame := anim.current_frame + 1
        if next_frame >= len(anim.frames) {
            if anim.loops {
                anim.current_frame = 0
            } else {
                anim.current_frame = len(anim.frames) - 1
                anim.state = .Stopped
                return true
            }
        } else {
            anim.current_frame = next_frame
        }
    }

    return false
}

get_current_frame :: proc(anim: ^Animation) -> Image_Id {
    if anim == nil {
        return .nil
    }

    if len(anim.frames) == 0 {
        return .nil
    }

    if anim.current_frame < 0 || anim.current_frame >= len(anim.frames) {
        return .nil
    }

    frame := anim.frames[anim.current_frame]
    return frame
}

draw_animated_sprite :: proc(pos: Vector2, anim: ^Animation, pivot := Pivot.bottom_left, xform := Matrix4(1), color_override := v4{0,0,0,0}, z_layer := ZLayer.nil) {
    if anim == nil do return
    current_frame := get_current_frame(anim)
    draw_sprite(pos, current_frame, pivot, xform, color_override, z_layer)
}

play_animation :: proc(anim: ^Animation){
    if anim == nil do return
    anim.state = .Playing
}

pause_animation :: proc(anim: ^Animation){
    if anim == nil do return
    anim.state = .Paused
}

stop_animation :: proc(anim: ^Animation) {
    if anim == nil do return
    anim.state = .Stopped
    anim.current_frame = 0
    anim.frame_timer = 0
}

reset_animation :: proc(anim: ^Animation){
    if anim == nil do return
    anim.current_frame = 0
    anim.frame_timer = 0
}

create_animation_collection :: proc() -> Animation_Collection {
    return Animation_Collection{
        animations = make(map[string]Animation),
        current_animation = "",
    }
}

add_animation :: proc(collection: ^Animation_Collection, animation: Animation){
    collection.animations[animation.name] = animation
}

play_animation_by_name :: proc(collection: ^Animation_Collection, name: string) {
    if collection == nil {
        return
    }

    if collection.current_animation == name {
        return
    }

    if collection.current_animation != "" {
        if anim, ok := &collection.animations[collection.current_animation]; ok {
            stop_animation(anim)
        }
    }

    if anim, ok := &collection.animations[name]; ok {
        collection.current_animation = name
        play_animation(anim)
    } else {
        fmt.println("Animation not found:", name)
    }
}

reset_and_play_animation :: proc(collection: ^Animation_Collection, name: string, speed: f32 = 1.0) {
    if collection == nil do return

    if anim, ok := &collection.animations[name]; ok {
        anim.current_frame = 0
        anim.frame_timer = 0
        anim.state = .Playing
        anim.loops = false
        adjust_animation_to_speed(anim, speed)

        collection.current_animation = name
    } else {
        fmt.println("Animation not found:", name)
    }
}

update_current_animation :: proc(collection: ^Animation_Collection, delta_t: f32) {
    if collection.current_animation != "" {
        if anim, ok := &collection.animations[collection.current_animation]; ok {
            animation_finished := update_animation(anim, delta_t)
            if animation_finished {
                collection.current_animation = ""
            }
        }
    }
}

draw_current_animation :: proc(collection: ^Animation_Collection, pos: Vector2, pivot := Pivot.bottom_left, xform := Matrix4(1), color_override := v4{0,0,0,0}, z_layer := ZLayer.nil) {
    if collection == nil || collection.current_animation == "" {
        return
    }
    if anim, ok := &collection.animations[collection.current_animation]; ok {
        current_frame := get_current_frame(anim)
        draw_animated_sprite(pos, anim, pivot, xform, color_override, z_layer)
    }
}

load_animation_frames :: proc(directory: string, prefix: string) -> ([]Image_Id, bool) {
    frames: [dynamic]Image_Id
    frames.allocator = context.temp_allocator

    dir_handle, err := os.open(directory)
    if err != 0 {
        log_error("Failed to open directory:", directory)
        return nil, false
    }
    defer os.close(dir_handle)

    files, read_err := os.read_dir(dir_handle, 0)
    if read_err != 0 {
        log_error("Failed to read directory:", directory)
        return nil, false
    }

    for file in files {
        if !strings.has_prefix(file.name, prefix) do continue
        if !strings.has_suffix(file.name, ".png") do continue

        frame_name := strings.concatenate({prefix, "_", strings.trim_suffix(file.name, ".png")})

        frame_id: Image_Id
        switch frame_name {
            case: continue
        }

        append(&frames, frame_id)
    }

    if len(frames) == 0 {
        log_error("No frames found for animation:", prefix)
        return nil, false
    }

    return frames[:], true
}

//
// :collision

AABB :: Vector4

aabb_collide :: proc(a, b: Vector4) ->bool {
    return !(a.z < b.x || a.x > b.z || a.w < b.y || a.y > b.w)
}

aabb_collide_aabb :: proc(a: AABB, b: AABB) -> (bool, Vector2) {
	dx := (a.z + a.x) / 2 - (b.z + b.x) / 2;
	dy := (a.w + a.y) / 2 - (b.w + b.y) / 2;
	overlap_x := (a.z - a.x) / 2 + (b.z - b.x) / 2 - abs(dx);
	overlap_y := (a.w - a.y) / 2 + (b.w - b.y) / 2 - abs(dy);
	if overlap_x <= 0 || overlap_y <= 0 {
		return false, Vector2{};
	}
	penetration := Vector2{};
	if overlap_x < overlap_y {
		penetration.x = overlap_x if dx > 0 else -overlap_x;
	} else {
		penetration.y = overlap_y if dy > 0 else -overlap_y;
	}
	return true, penetration;
}

aabb_get_center :: proc(a: Vector4) -> Vector2 {
	min := a.xy;
	max := a.zw;
	return { min.x + 0.5 * (max.x-min.x), min.y + 0.5 * (max.y-min.y) };
}

aabb_make_with_pos :: proc(pos: Vector2, size: Vector2, pivot: Pivot) -> Vector4 {
	aabb := (Vector4){0,0,size.x,size.y};
	aabb = aabb_shift(aabb, pos - scale_from_pivot(pivot) * size);
	return aabb;
}

aabb_make_with_size :: proc(size: Vector2, pivot: Pivot) -> Vector4 {
	return aabb_make({}, size, pivot);
}

aabb_make :: proc{
	aabb_make_with_pos,
	aabb_make_with_size
}

aabb_shift :: proc(aabb: Vector4, amount: Vector2) -> Vector4 {
	return {aabb.x + amount.x, aabb.y + amount.y, aabb.z + amount.x, aabb.w + amount.y};
}

aabb_contains :: proc(aabb: Vector4, p: Vector2) -> bool {
	return (p.x >= aabb.x) && (p.x <= aabb.z) &&
           (p.y >= aabb.y) && (p.y <= aabb.w);
}

aabb_size :: proc(aabb: AABB) -> Vector2 {
	return { abs(aabb.x - aabb.z), abs(aabb.y - aabb.w) }
}

//
// :ui & control
UI_Config :: struct {
   skills: Skills_UI_Config,
   quests: Quest_UI_Config,
   upgrades: Upgrades_UI_Config,
   gold_display: struct {
        pos_x: f32,
        pos_y: f32,
        text_offset_x: f32,
        text_offset_y: f32,
        text_scale: f32,
        sprite_pos_x: f32,
        sprite_pos_y: f32,
        sprite_size_x: f32,
        sprite_size_y: f32,
        sprite: Image_Id,
    },
    spawner: struct {
        pos_x: f32,
        pos_y: f32,
        size_x: f32,
        size_y: f32,
        bounds_x: f32,
        bounds_y: f32,
        txt_pos_x: f32,
        txt_pos_y: f32,
        sprite: Image_Id,
    }
}

UI_Hot_Reload :: struct {
   config_path: string,
   last_modified_time: time.Time,
   config: UI_Config,
}

UI_CONSTANTS :: struct {
    MENU_TRANSITION_SPEED: f32,
    HOVER_SCALE_SPEED: f32,
    TOOLTIP_FADE_SPEED: f32,
    SKILL_BUTTON_SIZE: Vector2,
    SKILL_MENU_SIZE: Vector2,
    SKILL_ICON_SIZE: Vector2,
    NORMAL_SCALE: f32,
    HOVER_SCALE: f32,
    MENU_PADDING: f32,
    ITEM_SPACING: f32,
}

UI := UI_CONSTANTS{
    MENU_TRANSITION_SPEED = 10.0,
    HOVER_SCALE_SPEED = 15.0,
    TOOLTIP_FADE_SPEED = 8.0,
    SKILL_BUTTON_SIZE = {48, 48},
    SKILL_MENU_SIZE = {400, 500},
    SKILL_ICON_SIZE = {40, 40},
    NORMAL_SCALE = 1.0,
    HOVER_SCALE = 1.1,
    MENU_PADDING = 20.0,
    ITEM_SPACING = 10.0,
}

UI_Colors :: struct {
    background: Vector4,
    panel: Vector4,
    button: Vector4,
    button_hover: Vector4,
    text: Vector4,
    text_disabled: Vector4,
    xp_bar_bg: Vector4,
    xp_bar_fill: Vector4,
    tooltip_bg: Vector4,
}

Colors := UI_Colors{
    background = {0.1, 0.1, 0.1, 0.95},
    panel = {0.2, 0.2, 0.2, 0.9},
    button = {0.3, 0.3, 0.3, 1.0},
    button_hover = {0.4, 0.4, 0.4, 1.0},
    text = {1.0, 1.0, 1.0, 1.0},
    text_disabled = {0.6, 0.6, 0.6, 1.0},
    xp_bar_bg = {0.15, 0.15, 0.15, 1.0},
    xp_bar_fill = {0.0, 0.8, 0.2, 1.0},
    tooltip_bg = {0.05, 0.05, 0.05, 0.95},
}

BUTTON_WIDTH :: 200.0
BUTTON_HEIGHT :: 50.0

init_ui_state :: proc(gs: ^Game_State) {
    gs.ui = {
        skills_button_scale = UI.NORMAL_SCALE,
        quest_button_scale = UI.NORMAL_SCALE,
        skills_menu_alpha = 0,
        skills_tooltip_alpha = 0,
        skills_hover_tooltip_active = false,
        skills_hover_tooltip_skill = nil,
        skills_menu_pos = Vector2{0, 0},
        last_mouse_pos = Vector2{0, 0},
    }
}

update_ui_state :: proc(gs: ^Game_State, dt: f32) {
    mouse_pos := mouse_pos_in_world_space()

    button_pos := get_skill_button_pos()
    if is_point_in_rect(mouse_pos, button_pos, UI.SKILL_BUTTON_SIZE) {
        if key_just_pressed(.LEFT_MOUSE) {
            gs.skills_system.menu_open = !gs.skills_system.menu_open
            if gs.skills_system.menu_open {
                gs.quests_system.menu_open = false
                gs.upgrades_menu_open = false
            }
        }
    } else {
        animate_to_target_f32(&gs.ui.skills_button_scale, UI.NORMAL_SCALE, dt, UI.HOVER_SCALE_SPEED)
    }

    cfg := gs.ui_config.quests.button
    quest_button_pos := v2{cfg.pos_x, cfg.pos_y}

    if is_point_in_rect(mouse_pos, quest_button_pos, UI.SKILL_BUTTON_SIZE) {
        if key_just_pressed(.LEFT_MOUSE) {
            gs.quests_system.menu_open = !gs.quests_system.menu_open
            if gs.quests_system.menu_open {
                gs.skills_system.menu_open = false
                gs.upgrades_menu_open = false
            }
        }
    } else {
        animate_to_target_f32(&gs.ui.quest_button_scale, UI.NORMAL_SCALE, dt, UI.HOVER_SCALE_SPEED)
    }

    cfg_upgrades := gs.ui_config.upgrades.button
    upgrades_button_pos := v2{cfg_upgrades.pos_x, cfg_upgrades.pos_y}

    if is_point_in_rect(mouse_pos, upgrades_button_pos, UI.SKILL_BUTTON_SIZE) {
        if key_just_pressed(.LEFT_MOUSE) {
            gs.upgrades_menu_open = !gs.upgrades_menu_open
            if gs.upgrades_menu_open {
                gs.skills_system.menu_open = false
                gs.quests_system.menu_open = false
            }
        }
    } else {
        animate_to_target_f32(&gs.ui.upgrades_button_scale, UI.NORMAL_SCALE, dt, UI.HOVER_SCALE_SPEED)
    }

    target_alpha := gs.skills_system.menu_open ? 1.0 : 0.0
    animate_to_target_f32(&gs.ui.skills_menu_alpha, f32(target_alpha), dt, UI.MENU_TRANSITION_SPEED)

    if gs.ui.skills_hover_tooltip_active {
        animate_to_target_f32(&gs.ui.skills_tooltip_alpha, 1.0, dt, UI.TOOLTIP_FADE_SPEED)
    } else {
        animate_to_target_f32(&gs.ui.skills_tooltip_alpha, 0.0, dt, UI.TOOLTIP_FADE_SPEED)
    }

    gs.ui.last_mouse_pos = mouse_pos
}

draw_panel :: proc(pos, size: Vector2, color: Vector4, z_layer := ZLayer.ui) {
    draw_rect_aabb(
        pos - size * 0.5,
        size,
        col = color,
        z_layer = z_layer,
    )
}

draw_skill_tooltip :: proc(skill: ^Skill) {
   if skill == nil do return
   cfg := gs.ui_config.skills

   push_z_layer(.ui)

   size := v2{cfg.tooltip.size_x, cfg.tooltip.size_y}
   pos := v2{cfg.tooltip.offset_x, cfg.tooltip.offset_y}
   padding := v2{10, 10}

   draw_sprite_1024(
        pos,
        size,
        Image_Id.tooltip_bg,
        pivot = .center_center,
        z_layer = .ui
   )

   text_start := v2{cfg.tooltip.text_pos_x, cfg.tooltip.text_pos_y}
   line_height := f32(cfg.tooltip.line_spacing)

   draw_wrapped_text(
        text_start + v2{0, line_height},
        skill.description,
        Text_Bounds{width=230, height=100},
        scale = 1.0,
        z_layer = .ui,
    )

   bonus := calculate_skill_bonus(skill) * 100
   draw_text(
       text_start + v2{0, line_height * 2},
       fmt.tprintf("Current Bonus: +%.1f%%", bonus),
       z_layer = .ui
   )

   draw_text(
       text_start + v2{0, line_height * 3},
       fmt.tprintf("Level: %d", skill.level),
       z_layer = .ui
   )
}

has_active_dummy :: proc() -> bool {
    active_dummies := 0
    max_dummies := gs.upgrades_system.multiple_dummy_tiers[gs.upgrades_system.current_multiple_tier].max_dummies

    for &en in gs.entities {
        if .allocated in en.flags && en.kind == .dummy {
            active_dummies += 1
        }
    }

    if active_dummies < max_dummies && gs.upgrades_system.auto_spawn_unlocked {
        gs.upgrades_system.frames_since_last_spawn += 1
        if gs.upgrades_system.frames_since_last_spawn > 60 {
            for pos in DUMMY_POSITIONS[0:max_dummies] {
                position_occupied := false
                for &en in gs.entities {
                    if .allocated in en.flags && en.kind == .dummy {
                        if linalg.length(en.pos - pos) < 10 {
                            position_occupied = true
                            break
                        }
                    }
                }
                if !position_occupied {
                    spawn_dummy(pos)
                    gs.upgrades_system.frames_since_last_spawn = 0
                    break
                }
            }
        }
    }

    return active_dummies >= max_dummies
}

check_spawn_button :: proc() {
    active_dummies := 0
    for &en in gs.entities {
        if .allocated in en.flags && en.kind == .dummy {
            active_dummies += 1
        }
    }

    current_tier := &gs.upgrades_system.multiple_dummy_tiers[gs.upgrades_system.current_multiple_tier]
    if active_dummies >= current_tier.max_dummies {
        return
    }

    cfg := gs.ui_config.spawner
    bounds_act := v2{cfg.bounds_x, cfg.bounds_y}

    mouse_pos := mouse_pos_in_world_space()
    button_bounds := aabb_make(v2{cfg.pos_x, cfg.pos_y}, bounds_act, Pivot.center_center)
    hover := aabb_contains(button_bounds, mouse_pos)

    if hover && key_just_pressed(.LEFT_MOUSE) {
        for pos in DUMMY_POSITIONS[0:current_tier.max_dummies] {
            position_occupied := false
            for &en in gs.entities {
                if .allocated in en.flags && en.kind == .dummy {
                    if linalg.length(en.pos - pos) < 10 {
                        position_occupied = true
                        break
                    }
                }
            }
            if !position_occupied {
                spawn_dummy(pos)
                break
            }
        }
    }
}

//
// :systems
XP_CONSTANTS :: struct {
    BASE_XP_FROM_DUMMY: int,
    BASE_XP_BOOST: f32,
    XP_BOOST_PER_LEVEL: f32,
    MAX_LEVEL: int,
    LEVEL_XP_MULTIPLIER: f32,
}

XP :: XP_CONSTANTS{
    BASE_XP_FROM_DUMMY = 500,
    BASE_XP_BOOST = 0.10,
    XP_BOOST_PER_LEVEL = 0.10,
    MAX_LEVEL = 25,
    LEVEL_XP_MULTIPLIER = 1.2,
}

//
// :quests
QUEST_TICK_TIME :: 1.5

Quest_Menu_Type :: enum {
    normal,
    advanced,
}

Quest_Type :: enum {
    nil,
    gold_generation,
    gold_and_xp,
    formation_training,
    meditation_training,
    war_treasury,
    tactical_assessment,
    strategic_command,
}

Quest :: struct {
    type: Quest_Type,
    name: string,
    description: string,
    level: int,
    is_unlocked: bool,
    cooldown: f32,
    required_skill: Skill_Type,
    required_skill_levels: [5]int,
    gold_per_tick: [5]int,
    progress: f32,
    display_progress: f32,
    xp_per_tick: f32,
    xp_target_skill: Skill_Type,
    tactical_mode_active: bool,
    tactical_mode_timer: f32,
    tactical_mode_combo: int,
    command_rotation_index: int,
    command_timer: f32,
    command_active: bool,
}

Quests_System :: struct {
    quests: [dynamic]Quest,
    active_quest: ^Quest,
    timer: f32,
    menu_open: bool,
    rotation_pairs: [][2]Quest_Type,
    active_menu: Quest_Menu_Type,
}

Quest_UI_Config :: struct {
    menu: struct {
        pos_x: f32,
        pos_y: f32,
        size_x: f32,
        size_y: f32,
        background_sprite: Image_Id,
    },
    button: struct {
        pos_x: f32,
        pos_y: f32,
        size_x: f32,
        size_y: f32,
        sprite: Image_Id,
    },
    next_quest: struct {
        offset_x: f32,
        offset_y: f32,
        panel_size_x: f32,
        panel_size_y: f32,
        title_offset_x: f32,
        title_offset_y: f32,
        name_offset_x: f32,
        name_offset_y: f32,
        desc_offset_x: f32,
        desc_offset_y: f32,
        level_req_offset_x: f32,
        level_req_offset_y: f32,
    },
    switch_button: struct {
        offset_x: f32,
        offset_y: f32,
        size_x: f32,
        size_y: f32,
        bounds_x: f32,
        bounds_y: f32,
        text_scale: f32,
        sprite: Image_Id,
    },
    quest_entry: struct {
        start_offset_x: f32,
        start_offset_y: f32,
        spacing_y: f32,
        size_x: f32,
        size_y: f32,
        name_offset_x: f32,
        name_offset_y: f32,
        reward_offset_x: f32,
        reward_offset_y: f32,
        next_level_offset_x: f32,
        next_level_offset_y: f32,
        select_button: struct {
            offset_x: f32,
            offset_y: f32,
            size_x: f32,
            size_y: f32,
            text_scale: f32,
            text_pos_x: f32,
            text_pos_y: f32,
            bounds_x: f32,
            bounds_y: f32,
        },
        xp_bar: struct {
            text_pos_x: f32,
            text_pos_y: f32,
            text_scale: f32,
            text_layer: ZLayer,
            sprite: Image_Id,
            sprite_layer: ZLayer,
            sprite_width: f32,
            sprite_height: f32,
            sprite_pos_x: f32,
            sprite_pos_y: f32,
            bar_layer: ZLayer,
            bar_width: f32,
            bar_height: f32,
            bar_pos_x: f32,
            bar_pos_y: f32,
        },
    },
}

init_quests_system :: proc() -> Quests_System {
    system := Quests_System {
        quests = make([dynamic]Quest),
        active_quest = nil,
        timer = 0,
        menu_open = false,
        active_menu = .normal,
        rotation_pairs = make([][2]Quest_Type, 6),
    }

    apprentice_quest := Quest{
        type = .gold_generation,
        name = "Apprentice's Dedication",
        description = "Study ancient scrolls and practice basic archery techniques",
        level = 1,
        is_unlocked = false,
        cooldown = QUEST_TICK_TIME,
        required_skill = .xp_boost,
        required_skill_levels = {4, 5, 6, 7, 8},
        gold_per_tick = {1, 2, 3, 4, 5},
    }

    warrior_quest := Quest{
        type = .gold_generation,
        name = "Warrior's Resolve",
        description = "Train with weighted bows and reinforced targets",
        level = 1,
        is_unlocked = false,
        cooldown = QUEST_TICK_TIME,
        required_skill = .strength_boost,
        required_skill_levels = {6, 7, 8, 9, 10},
        gold_per_tick = {3, 6, 9, 12, 15},
    }

    wind_walker_quest := Quest{
        type = .gold_and_xp,
        name = "Wind Walker's Grace",
        description = "Practice quick-draw techniques with master archers",
        level = 1,
        is_unlocked = false,
        cooldown = QUEST_TICK_TIME,
        required_skill = .speed_boost,
        required_skill_levels = {10, 12, 14, 16, 18},
        gold_per_tick = {9, 18, 27, 80, 160},
        xp_per_tick = 15,
        xp_target_skill = .xp_boost,
    }

    hunters_precision_quest := Quest{
        type = .gold_and_xp,
        name = "Hunter's Precision",
        description = "Perfect vital point targeting on moving targets",
        level = 1,
        is_unlocked = false,
        cooldown = QUEST_TICK_TIME,
        required_skill = .critical_boost,
        required_skill_levels = {20, 22, 24, 26, 28},
        gold_per_tick = {20, 40, 60, 80, 100},
        xp_per_tick = 15,
        xp_target_skill = .strength_boost,
    }

    multi_shot_quest := Quest{
        type = .gold_and_xp,
        name = "Multi-Shot",
        description = "Master the art of rapid arrow nocking",
        level = 1,
        is_unlocked = false,
        cooldown = QUEST_TICK_TIME,
        required_skill = .storm_arrows,
        required_skill_levels = {40, 44, 48, 52, 56},
        gold_per_tick = {40, 80, 120, 160, 200},
        xp_per_tick = 15,
        xp_target_skill = .speed_boost,
    }

    elemental_archery_quest := Quest{
        type = .gold_and_xp,
        name = "Elemental Archery",
        description = "Study with the realm's enchanted bowyers",
        level = 1,
        is_unlocked = false,
        cooldown = QUEST_TICK_TIME,
        required_skill = .mystic_fletcher,
        required_skill_levels = {48, 56, 64, 72, 80},
        gold_per_tick = {80, 160, 240, 320, 400},
        xp_per_tick = 15,
        xp_target_skill = .storm_arrows,
    }

    battle_recovery_quest := Quest{
        type = .gold_and_xp,
        name = "Battle Recovery",
        description = "Endurance training with veteran rangers",
        level = 1,
        is_unlocked = false,
        cooldown = QUEST_TICK_TIME,
        required_skill = .warrior_stamina,
        required_skill_levels = {56, 64, 72, 80, 88},
        gold_per_tick = {160, 240, 320, 400, 480},
        xp_per_tick = 15,
        xp_target_skill = .mystic_fletcher,
    }

    strategic_positioning_quest := Quest{
        type = .formation_training,
        name = "Strategic Positioning",
        description = "Train squad formations with a chance to rapidly improve random skills",
        level = 1,
        is_unlocked = false,
        cooldown = QUEST_TICK_TIME,
        required_skill = .formation_mastery,
        required_skill_levels = {62, 70, 78, 86, 94},
        gold_per_tick = {200, 300, 400, 500, 600},
    }

    inner_focus_quest := Quest{
        type = .meditation_training,
        name = "Inner Focus",
        description = "Deep meditation rituals to extend your mental focus",
        level = 1,
        is_unlocked = false,
        cooldown = QUEST_TICK_TIME,
        required_skill = .battle_meditation,
        required_skill_levels = {80, 86, 92, 98, 104},
        gold_per_tick = {240, 340, 440, 540, 640},
    }

    war_treasury_quest := Quest{
        type = .war_treasury,
        name = "War Treasury",
        description = "Master the art of resource management in battle",
        level = 1,
        is_unlocked = false,
        cooldown = QUEST_TICK_TIME,
        required_skill = .war_preparation,
        required_skill_levels = {100, 110, 120, 130, 140},
        gold_per_tick = {260, 380, 500, 620, 740},
    }

    tactical_assessment_quest := Quest{
        type = .tactical_assessment,
        name = "Tactical Assessment",
        description = "Study enemy vulnerabilities to enhance combat effectiveness",
        level = 1,
        is_unlocked = false,
        cooldown = QUEST_TICK_TIME,
        required_skill = .tactical_analysis,
        required_skill_levels = {110, 120, 130, 140, 150},
        gold_per_tick = {380, 480, 580, 680, 780},
        tactical_mode_active = false,
        tactical_mode_timer = 0,
        tactical_mode_combo = 0,
    }

    strategic_command_quest := Quest{
        type = .strategic_command,
        name = "Strategic Command",
        description = "Coordinate multiple training operations simultaneously",
        level = 1,
        is_unlocked = false,
        cooldown = QUEST_TICK_TIME,
        required_skill = .commanders_authority,
        required_skill_levels = {120, 140, 160, 180, 200},
        gold_per_tick = {500, 600, 700, 800, 900},
        command_rotation_index = 0,
        command_timer = 0,
        command_active = false,
    }


    // NORMAL
    append(&system.quests, apprentice_quest)
    append(&system.quests, warrior_quest)
    append(&system.quests, wind_walker_quest)
    append(&system.quests, hunters_precision_quest)
    append(&system.quests, multi_shot_quest)
    append(&system.quests, elemental_archery_quest)
    append(&system.quests, battle_recovery_quest)

    // ADVANCED
    append(&system.quests, strategic_positioning_quest)
    append(&system.quests, inner_focus_quest)
    append(&system.quests, war_treasury_quest)
    append(&system.quests, tactical_assessment_quest)
    append(&system.quests, strategic_command_quest)

    system.rotation_pairs = make([][2]Quest_Type, 6)
    system.rotation_pairs[0] = {.gold_generation, .gold_and_xp}
    system.rotation_pairs[1] = {.meditation_training, .formation_training}
    system.rotation_pairs[2] = {.war_treasury, .tactical_assessment}
    system.rotation_pairs[3] = {.gold_and_xp, .formation_training}
    system.rotation_pairs[4] = {.gold_generation, .tactical_assessment}
    system.rotation_pairs[5] = {.war_treasury, .meditation_training}

    return system
}

update_quests_system :: proc(system: ^Quests_System, dt: f32) {
    if system.active_quest == nil do return

    system.timer -= dt
    if system.timer <= 0 {
        system.timer = QUEST_TICK_TIME
        give_quest_rewards(system.active_quest)
        system.active_quest.progress = 0
    } else {
        system.active_quest.progress = 1 - (system.timer / QUEST_TICK_TIME)
    }

    if system.active_quest.type == .tactical_assessment && system.active_quest.tactical_mode_active {
        system.active_quest.tactical_mode_timer -= dt
        if system.active_quest.tactical_mode_timer <= 0 {
            system.active_quest.tactical_mode_active = false
            system.active_quest.tactical_mode_combo = 0

            add_floating_text_params(
                v2{-200, 150},
                "Tactical Assessment Ended",
                v4{0.8, 0.3, 0.3, 0.7},
                scale = 0.8,
                target_scale = 1.0,
                lifetime = 1.0,
                velocity = v2{0, 75},
            )
        }
    }

    if system.active_quest != nil &&
       system.active_quest.type == .strategic_command &&
       system.active_quest.command_active {

        system.active_quest.command_timer -= dt
        if system.active_quest.command_timer <= 0 {
            system.active_quest.command_active = false

            add_floating_text_params(
                v2{-200, 150},
                "Command Order Ended",
                v4{0.4, 0.4, 0.8, 0.7},
                scale = 0.8,
                target_scale = 1.0,
                lifetime = 1.0,
                velocity = v2{0, 75},
            )
        }
    }

    if system.active_quest != nil {
        animate_to_target_f32(
            &system.active_quest.display_progress,
            system.active_quest.progress,
            dt * 5
        )
    }
}

give_quest_rewards :: proc(quest: ^Quest) {
    if quest == nil do return

    #partial switch quest.type {
        case .gold_generation:
            base_gold := quest.gold_per_tick[quest.level - 1]
            final_gold := calculate_gold_gain(base_gold)
            gs.skills_system.gold += final_gold

            cfg := gs.ui_config.gold_display
            gold_pos := v2{cfg.pos_x + cfg.text_offset_x + 100, cfg.pos_y + cfg.text_offset_y}
            add_floating_text_params(
                gold_pos,
                fmt.tprintf("+%d Gold", final_gold),
                v4{1,0.8,0,1},
                scale = 0.7,
                target_scale = 0.8,
                lifetime = 1.0,
                velocity = v2{0, 50},
            )
        case .gold_and_xp:
            base_gold := quest.gold_per_tick[quest.level - 1]
            final_gold := calculate_gold_gain(base_gold)
            gs.skills_system.gold += final_gold

            cfg := gs.ui_config.gold_display
            gold_pos := v2{cfg.pos_x + cfg.text_offset_x + 100, cfg.pos_y + cfg.text_offset_y}
            add_floating_text_params(
                gold_pos,
                fmt.tprintf("+%d Gold", final_gold),
                v4{1,0.8,0,1},
                scale = 0.7,
                target_scale = 0.8,
                lifetime = 1.0,
                velocity = v2{0, 50},
            )

            for &skill in gs.skills_system.skills {
                if skill.is_unlocked && skill.type == quest.xp_target_skill {
                    if skill.level < XP.MAX_LEVEL {
                        prev_level := skill.level
                        skill.current_xp += int(quest.xp_per_tick)

                        for skill.current_xp >= skill.xp_to_next_level {
                            skill.current_xp -= skill.xp_to_next_level
                            skill.level += 1

                            if skill.level >= XP.MAX_LEVEL {
                                skill.level = XP.MAX_LEVEL
                                skill.current_xp = 0
                                break
                            }

                            skill.xp_to_next_level = int(f32(skill.xp_to_next_level) * XP.LEVEL_XP_MULTIPLIER)
                        }

                        if skill.level > prev_level {
                            check_quest_unlocks(&skill)
                            add_floating_text_params(
                                v2{-200, 150},
                                fmt.tprintf("%s reached level %d!", skill.name, skill.level),
                                v4{0.5, 0.5, 1, 1},
                                scale = 0.8,
                                target_scale = 1.0,
                                lifetime = 1.0,
                                velocity = v2{0, 50},
                            )
                        }

                        xp_pos := gold_pos + v2{0, 30}
                        add_floating_text_params(
                            xp_pos,
                            fmt.tprintf("+%d XP to %s", int(quest.xp_per_tick), skill.name),
                            v4{0.5, 0.8, 1, 1},
                            scale = 0.7,
                            target_scale = 0.8,
                            lifetime = 1.0,
                            velocity = v2{0, 50},
                        )
                    }
                    break
                }
            }
        case .formation_training:
            base_gold := quest.gold_per_tick[quest.level - 1]
            final_gold := calculate_gold_gain(base_gold)
            gs.skills_system.gold += final_gold

            cfg := gs.ui_config.gold_display
            gold_pos := v2{cfg.pos_x + cfg.text_offset_x + 100, cfg.pos_y + cfg.text_offset_y}
            add_floating_text_params(
                gold_pos,
                fmt.tprintf("+%d Gold", final_gold),
                v4{1,0.8,0,1},
                scale = 0.7,
                target_scale = 0.8,
                lifetime = 1.0,
                velocity = v2{0, 50},
            )

            trigger_chance := 0.02 * f32(quest.level)

            if rand.float32() < trigger_chance {
                unlocked_skills: [dynamic]^Skill
                defer delete(unlocked_skills)

                for &skill in gs.skills_system.skills {
                    if skill.is_unlocked && skill.level < XP.MAX_LEVEL {
                        append(&unlocked_skills, &skill)
                    }
                }
                for &skill in gs.skills_system.advanced_skills {
                    if skill.is_unlocked && skill.level < XP.MAX_LEVEL {
                        append(&unlocked_skills, &skill)
                    }
                }

                if len(unlocked_skills) > 0 {
                    random_skill := unlocked_skills[rand.int_max(len(unlocked_skills))]

                    xp_boost := int(f32(random_skill.xp_to_next_level) * 0.25)

                    prev_level := random_skill.level
                    random_skill.current_xp += xp_boost

                    for random_skill.current_xp >= random_skill.xp_to_next_level {
                        random_skill.current_xp -= random_skill.xp_to_next_level
                        random_skill.level += 1

                        if random_skill.level >= XP.MAX_LEVEL {
                            random_skill.level = XP.MAX_LEVEL
                            random_skill.current_xp = 0
                            break
                        }

                        random_skill.xp_to_next_level = int(f32(random_skill.xp_to_next_level) * XP.LEVEL_XP_MULTIPLIER)
                    }

                    add_floating_text_params(
                        v2{-200, 150},
                        fmt.tprintf("Formation Mastery: +%d XP to %s!", xp_boost, random_skill.name),
                        v4{0.7, 0.3, 1.0, 1.0},
                        scale = 0.8,
                        target_scale = 1.0,
                        lifetime = 1.0,
                        velocity = v2{0, 75},
                    )

                    if random_skill.level > prev_level {
                        check_quest_unlocks(random_skill)
                        add_floating_text_params(
                            v2{-200, 200},
                            fmt.tprintf("%s reached level %d!", random_skill.name, random_skill.level),
                            v4{1, 0.5, 1, 1},
                            scale = 0.8,
                            target_scale = 1.0,
                            lifetime = 1.0,
                            velocity = v2{0, 75},
                        )
                    }
                }
            }
        case .meditation_training:
            base_gold := quest.gold_per_tick[quest.level - 1]
            final_gold := calculate_gold_gain(base_gold)
            gs.skills_system.gold += final_gold

            cfg := gs.ui_config.gold_display
            gold_pos := v2{cfg.pos_x + cfg.text_offset_x + 100, cfg.pos_y + cfg.text_offset_y}
            add_floating_text_params(
                gold_pos,
                fmt.tprintf("+%d Gold", final_gold),
                v4{1, 0.8, 0, 1},
                scale = 0.7,
                target_scale = 0.8,
                lifetime = 1.0,
                velocity = v2{0, 50},
            )
        case .war_treasury:
            base_gold := quest.gold_per_tick[quest.level - 1]
            final_gold := calculate_gold_gain(base_gold)
            gs.skills_system.gold += final_gold

            cfg := gs.ui_config.gold_display
            gold_pos := v2{cfg.pos_x + cfg.text_offset_x + 100, cfg.pos_y + cfg.text_offset_y}
            add_floating_text_params(
                gold_pos,
                fmt.tprintf("+%d Gold", final_gold),
                v4{1, 0.8, 0, 1},
                scale = 0.7,
                target_scale = 0.8,
                lifetime = 1.0,
                velocity = v2{0, 50},
            )

            trigger_chance := 0.05 + 0.05 * f32(quest.level - 1)
            if rand.float32() < trigger_chance {
                bonus_multiplier := 0.5 + 0.1 * f32(quest.level - 1)
                bonus_gold := int(f32(final_gold) * bonus_multiplier)
                gs.skills_system.gold += bonus_gold

                bonus_pos := gold_pos + v2{0, 10}
                add_floating_text_params(
                    bonus_pos,
                    fmt.tprintf("Treasury Insight: +%d Gold!", bonus_gold),
                    v4{1, 0.9, 0.2, 1},
                    scale = 1.0,
                    target_scale = 1.3,
                    lifetime = 1.5,
                    velocity = v2{0, 75},
                )
            }
        case .tactical_assessment:
            base_gold := quest.gold_per_tick[quest.level - 1]
            final_gold := calculate_gold_gain(base_gold)
            gs.skills_system.gold += final_gold

            cfg := gs.ui_config.gold_display
            gold_pos := v2{cfg.pos_x + cfg.text_offset_x + 100, cfg.pos_y + cfg.text_offset_y}
            add_floating_text_params(
                gold_pos,
                fmt.tprintf("+%d Gold", final_gold),
                v4{1, 0.8, 0, 1},
                scale = 0.7,
                target_scale = 0.8,
                lifetime = 1.0,
                velocity = v2{0, 50},
            )

            trigger_chance := 0.15 + 0.05 * f32(quest.level - 1)
            if rand.float32() < trigger_chance && !quest.tactical_mode_active {
                quest.tactical_mode_active = true
                quest.tactical_mode_timer = 5.0 + f32(quest.level - 1)
                quest.tactical_mode_combo = 0

                add_floating_text_params(
                    v2{-200, 150},
                    "Tactical Assessment Active!",
                    v4{0.8, 0.3, 0.3, 1.0},
                    scale = 0.8,
                    target_scale = 1.0,
                    lifetime = 1.0,
                    velocity = v2{0, 75},
                )
            }
        case .strategic_command:
            base_gold := quest.gold_per_tick[quest.level - 1]
            final_gold := calculate_gold_gain(base_gold)
            gs.skills_system.gold += final_gold

            cfg := gs.ui_config.gold_display
            gold_pos := v2{cfg.pos_x + cfg.text_offset_x + 100, cfg.pos_y + cfg.text_offset_y}
            add_floating_text_params(
                gold_pos,
                fmt.tprintf("+%d Gold", final_gold),
                v4{1,0.8,0,1},
                scale = 0.7,
                target_scale = 0.8,
                lifetime = 1.0,
                velocity = v2{0, 50},
            )

            trigger_chance := 0.15 + 0.05 * f32(quest.level - 1)
            if rand.float32() < trigger_chance && !quest.command_active {
                quest1, quest2 := get_next_quest_pair(quest, &gs.quests_system)
                if quest1 != nil && quest2 != nil {
                    quest.command_active = true
                    quest.command_timer = 3.0

                    reward_scale := 0.7

                    old_gold := gs.skills_system.gold
                    give_quest_rewards(quest1)
                    gold_diff1 := gs.skills_system.gold - old_gold
                    gs.skills_system.gold = old_gold + int(f32(gold_diff1) * f32(reward_scale))

                    old_gold = gs.skills_system.gold
                    give_quest_rewards(quest2)
                    gold_diff2 := gs.skills_system.gold - old_gold
                    gs.skills_system.gold = old_gold + int(f32(gold_diff2) * f32(reward_scale))

                    add_floating_text_params(
                        v2{-200, 150},
                        fmt.tprintf("Command Order: %s + %s", quest1.name, quest2.name),
                        v4{0.4, 0.4, 0.8, 1.0},
                        scale = 0.8,
                        target_scale = 1.0,
                        lifetime = 1.5,
                        velocity = v2{0, 75},
                    )

                    next_pair := rotation_pairs[quest.command_rotation_index]
                    add_floating_text_params(
                        v2{-200, 100},
                        fmt.tprintf("Next: %s + %s",
                            get_quest_type_name(next_pair[0]),
                            get_quest_type_name(next_pair[1])),
                        v4{0.4, 0.4, 0.8, 0.8},
                        scale = 0.7,
                        target_scale = 0.8,
                        lifetime = 1.0,
                        velocity = v2{0, 50},
                    )
                }
            }

    }
}

get_quest_type_name :: proc(type: Quest_Type) -> string {
    #partial switch type {
        case .gold_generation: return "Gold Training"
        case .gold_and_xp: return "Advanced Training"
        case .formation_training: return "Formation"
        case .meditation_training: return "Meditation"
        case .war_treasury: return "Treasury"
        case .tactical_assessment: return "Tactical"
        case: return "Unknown"
    }
}

check_quest_unlocks :: proc(skill: ^Skill) {
    if skill == nil do return

    for &quest in gs.quests_system.quests {
        if !quest.is_unlocked && quest.required_skill == skill.type {
            if skill.level >= quest.required_skill_levels[0] {
                quest.is_unlocked = true
            }
        }

        if quest.is_unlocked && quest.required_skill == skill.type {
            for level := len(quest.required_skill_levels)-1; level >= 0; level -= 1 {
                if skill.level >= quest.required_skill_levels[level] {
                    if quest.level < level + 1 {
                        old_level := quest.level
                        quest.level = level + 1

                        add_floating_text_params(
                            v2{-200, 150},
                            fmt.tprintf("%s advanced to level %d!", quest.name, quest.level),
                            v4{1, 0.8, 0, 1},
                            scale = 0.8,
                            target_scale = 1.0,
                            lifetime = 1.0,
                            velocity = v2{0, 50},
                        )
                    }
                    break
                }
            }
        }
    }
}

render_quests_ui :: proc() {
    if !gs.quests_system.menu_open do return

    cfg := gs.ui_config.quests
    push_z_layer(.ui)

    menu_pos := v2{cfg.menu.pos_x, cfg.menu.pos_y}
    menu_size := v2{cfg.menu.size_x, cfg.menu.size_y}

    draw_sprite_with_size(
        menu_pos,
        menu_size,
        cfg.menu.background_sprite,
        pivot = .center_center,
        z_layer = .ui,
    )

    button_pos := menu_pos + v2{cfg.switch_button.offset_x, cfg.switch_button.offset_y}
    button_size := v2{cfg.switch_button.size_x, cfg.switch_button.size_y}
    btn_bounds := v2{cfg.switch_button.bounds_x, cfg.switch_button.bounds_y}
    hover := is_point_in_rect(mouse_pos_in_world_space(), button_pos, btn_bounds)

    draw_sprite_with_size(
        button_pos,
        button_size,
        img_id = !hover ? Image_Id.next_skill_button_bg : Image_Id.next_skill_button_active_bg,
        pivot = .center_center,
        z_layer = .ui,
    )

    button_text := gs.quests_system.active_menu == .normal ? "Advanced" : "Normal"
    draw_text(
        button_pos,
        button_text,
        scale = auto_cast cfg.switch_button.text_scale,
        pivot = .center_center,
        z_layer = .ui,
    )

    if hover && key_just_pressed(.LEFT_MOUSE) {
        gs.quests_system.active_menu = gs.quests_system.active_menu == .normal ? .advanced : .normal
    }

    if gs.quests_system.active_menu == .normal {
        draw_normal_quests_content(menu_pos)
    } else {
        draw_advanced_quests_content(menu_pos)
    }
}


render_quest_entry_configured :: proc(quest: ^Quest, pos: Vector2) {
    cfg := gs.ui_config.quests.quest_entry
    is_active := gs.quests_system.active_quest == quest
    bg_color := is_active ? v4{0.3, 0.3, 0.3, 0.8} : v4{0.2, 0.2, 0.2, 0.8}

    entry_size := v2{cfg.size_x, cfg.size_y}
    draw_rect_aabb(pos, entry_size, col = bg_color, z_layer = .ui)

    name_pos := pos + v2{cfg.name_offset_x, cfg.name_offset_y}
    draw_text(name_pos, fmt.tprintf("%s (Level %d)", quest.name, quest.level), z_layer = .ui)

    reward_pos := pos + v2{cfg.reward_offset_x, cfg.reward_offset_y}
    current_gold := quest.gold_per_tick[quest.level-1]

    if quest.level < len(quest.gold_per_tick) {
        next_level_pos := pos + v2{cfg.next_level_offset_x, cfg.next_level_offset_y}
        required_skill_level := quest.required_skill_levels[quest.level]
        draw_text(
            next_level_pos,
            fmt.tprintf("Next Level: %d gold (Requires skill level %d)",
                quest.gold_per_tick[quest.level],
                required_skill_level),
            scale = 1.0,
            z_layer = .ui,
        )
    }

    draw_text(reward_pos, fmt.tprintf("Current: %d gold", current_gold), z_layer = .ui)

    if is_active { // XP BAR
        sprite_width := cfg.xp_bar.sprite_width
        sprite_height := cfg.xp_bar.sprite_height
        sprite_pos := pos + v2{cfg.xp_bar.sprite_pos_x, cfg.xp_bar.sprite_pos_y}
        bar_width := cfg.xp_bar.bar_width
        bar_height := cfg.xp_bar.bar_height
        bar_pos := pos + v2{cfg.xp_bar.bar_pos_x, cfg.xp_bar.bar_pos_y}

        draw_sprite_with_size(
            sprite_pos,
            v2{sprite_width, sprite_height},
            cfg.xp_bar.sprite,
            z_layer = cfg.xp_bar.sprite_layer,
        )

        progress_ratio := quest.display_progress
        draw_rect_aabb(
            bar_pos,
            v2{bar_width, bar_height},
            col = Colors.xp_bar_bg,
            z_layer = cfg.xp_bar.bar_layer
        )

        draw_rect_aabb(
            bar_pos,
            v2{bar_width * progress_ratio, bar_height},
            col = Colors.xp_bar_fill,
            z_layer = cfg.xp_bar.bar_layer
        )
    } else { // SELECT BUTTON
        button_pos := pos + v2{cfg.select_button.offset_x, cfg.select_button.offset_y}
        button_size := v2{cfg.select_button.size_x, cfg.select_button.size_y}
        btn_bounds := v2{cfg.select_button.bounds_x, cfg.select_button.bounds_y}

        mouse_pos := mouse_pos_in_world_space()
        hover := is_point_in_rect(mouse_pos_in_world_space(), button_pos, btn_bounds)

        draw_sprite_with_size(
            button_pos,
            button_size,
            img_id = !hover ? Image_Id.next_skill_button_bg : Image_Id.next_skill_button_active_bg,
            pivot = .center_center,
            z_layer = .ui,
        )

        text_pos := button_pos + v2{cfg.select_button.text_pos_x, cfg.select_button.text_pos_y}
        text_scale := cfg.select_button.text_scale
        draw_text(
            text_pos,
            scale = auto_cast text_scale,
            text = "Select",
            pivot = .center_center,
            z_layer = .ui
        )

        if hover && key_just_pressed(.LEFT_MOUSE) {
            gs.quests_system.active_quest = quest
            gs.quests_system.timer = QUEST_TICK_TIME
            quest.progress = 0
            quest.display_progress = 0
        }
    }
}

draw_normal_quests_content :: proc(menu_pos: Vector2) {
    cfg := gs.ui_config.quests

    next_quest: ^Quest
    for &quest in gs.quests_system.quests {
        if !quest.is_unlocked && is_normal_quest(quest.type) {
            next_quest = &quest
            break
        }
    }

    if next_quest != nil {
        next_quest_pos := menu_pos + v2{cfg.next_quest.offset_x, cfg.next_quest.offset_y}
        draw_next_quest_panel(next_quest, next_quest_pos)
    }

    entry_cfg := cfg.quest_entry
    quest_y := menu_pos.y + entry_cfg.start_offset_y

    for &quest in gs.quests_system.quests {
        if quest.is_unlocked && is_normal_quest(quest.type) {
            render_quest_entry_configured(&quest, v2{menu_pos.x + entry_cfg.start_offset_x, quest_y})
            quest_y -= entry_cfg.spacing_y
        }
    }
}

draw_advanced_quests_content :: proc(menu_pos: Vector2) {
    cfg := gs.ui_config.quests

    next_quest: ^Quest
    for &quest in gs.quests_system.quests {
        if !quest.is_unlocked && is_advanced_quest(quest.type) {
            next_quest = &quest
            break
        }
    }

    if next_quest != nil {
        next_quest_pos := menu_pos + v2{cfg.next_quest.offset_x, cfg.next_quest.offset_y}
        draw_next_quest_panel(next_quest, next_quest_pos)
    }

    entry_cfg := cfg.quest_entry
    quest_y := menu_pos.y + entry_cfg.start_offset_y

    for &quest in gs.quests_system.quests {
        if quest.is_unlocked && is_advanced_quest(quest.type) {
            render_quest_entry_configured(&quest, v2{menu_pos.x + entry_cfg.start_offset_x, quest_y})
            quest_y -= entry_cfg.spacing_y
        }
    }
}

is_normal_quest :: proc(type: Quest_Type) -> bool {
    #partial switch type {
        case .gold_generation, .gold_and_xp:
            return true
    }
    return false
}

is_advanced_quest :: proc(type: Quest_Type) -> bool {
    #partial switch type {
        case .formation_training, .meditation_training, .war_treasury,
             .tactical_assessment, .strategic_command:
            return true
    }
    return false
}

render_active_quest_ui :: proc(quest: ^Quest) {
    if quest == nil do return

    cfg := gs.ui_config.quests.quest_entry

    text_pos := v2{cfg.xp_bar.text_pos_x, cfg.xp_bar.text_pos_y}
    draw_text(
        text_pos,
        fmt.tprintf("%s (Level %d)", quest.name, quest.level),
        scale = f64(cfg.xp_bar.text_scale),
        z_layer = cfg.xp_bar.text_layer
    )

    sprite_width := cfg.xp_bar.sprite_width
    sprite_height := cfg.xp_bar.sprite_height
    sprite_size := v2{sprite_width, sprite_height}
    sprite_pos := text_pos + v2{cfg.xp_bar.sprite_pos_x, cfg.xp_bar.sprite_pos_y}

    bar_width := cfg.xp_bar.bar_width
    bar_height := cfg.xp_bar.bar_height
    bar_size := v2{bar_width, bar_height}
    bar_pos := text_pos + v2{cfg.xp_bar.bar_pos_x, cfg.xp_bar.bar_pos_y}

    draw_sprite_with_size(
        sprite_pos,
        sprite_size,
        cfg.xp_bar.sprite,
        z_layer = cfg.xp_bar.sprite_layer,
    )

    draw_rect_aabb(
        bar_pos,
        bar_size,
        col = Colors.xp_bar_fill,
        z_layer = cfg.xp_bar.bar_layer,
    )

    reward_pos := bar_pos + v2{0, -20}
    draw_text(
        reward_pos,
        fmt.tprintf("Gold per tick: %d", quest.gold_per_tick[quest.level - 1]),
        z_layer = .xp_bars
    )
}

render_quest_menu_button :: proc() {
    cfg := gs.ui_config.quests.button
    button_pos := v2{cfg.pos_x, cfg.pos_y}
    button_size := v2{cfg.size_x, cfg.size_y} * gs.ui.quest_button_scale

    draw_sprite_with_size(
        button_pos,
        button_size,
        cfg.sprite,
        pivot = .center_center,
        xform = xform_scale(v2{gs.ui.quest_button_scale, gs.ui.quest_button_scale}),
        z_layer = .ui,
    )
}

draw_next_quest_panel :: proc(quest: ^Quest, pos: Vector2) {
    if quest == nil do return

    first_unlockable: ^Quest
    for &q in gs.quests_system.quests {
        if !q.is_unlocked {
            first_unlockable = &q
            break
        }
    }

    if first_unlockable == nil do return

    required_skill_name := ""
    for skill in gs.skills_system.skills {
        if skill.type == first_unlockable.required_skill {
            required_skill_name = skill.name
            break
        }
    }

    cfg := gs.ui_config.quests.next_quest
    push_z_layer(.ui)
    draw_sprite(pos,
        .next_skill_panel_bg,
        pivot = .center_center,
        color_override = v4{1,1,1,0},
        z_layer = .ui
    )

    title_pos := pos + v2{cfg.title_offset_x, cfg.title_offset_y}
    draw_text(
        title_pos,
        "Next Available Quest",
        col = Colors.text * v4{1,1,1,1},
        scale = 1.2,
        pivot = .center_left,
        z_layer = .ui,
    )

    name_pos := pos + v2{cfg.name_offset_x, cfg.name_offset_y}
    draw_text(
        name_pos,
        first_unlockable.name,
        col = Colors.text * v4{1,1,1,1},
        pivot = .center_left,
        z_layer = .ui
    )

    desc_pos := pos + v2{cfg.desc_offset_x, cfg.desc_offset_y}
    draw_wrapped_text(
        desc_pos,
        first_unlockable.description,
        Text_Bounds{width=230, height=100},
        pivot = .center_left,
        z_layer = .ui
    )

    required_skill := first_unlockable.required_skill
    required_level := first_unlockable.required_skill_levels[0]

    level_pos := pos + v2{cfg.level_req_offset_x, cfg.level_req_offset_y}
    draw_text(
        level_pos,
        fmt.tprintf("Requires %s Level %d", required_skill_name, first_unlockable.required_skill_levels[0]),
        col = Colors.text * v4{1,1,1,1},
        pivot = .center_left,
        z_layer = .ui
    )
}

has_unlocked_quests :: proc(system: ^Quests_System) -> bool {
    for quest in system.quests {
        if quest.is_unlocked {
            return true
        }
    }

    return false
}

rotation_pairs := [][2]Quest_Type{
    {.gold_generation, .gold_and_xp},
    {.meditation_training, .formation_training},
    {.war_treasury, .tactical_assessment},
    {.gold_and_xp, .formation_training},
    {.gold_generation, .tactical_assessment},
    {.war_treasury, .meditation_training},
}

get_next_quest_pair :: proc(quest: ^Quest, system: ^Quests_System) -> (^Quest, ^Quest) {
    current_pair := rotation_pairs[quest.command_rotation_index]
    quest.command_rotation_index = (quest.command_rotation_index + 1) % len(rotation_pairs)

    quest1, quest2: ^Quest
    for &q in system.quests {
        if q.is_unlocked && q.type == current_pair[0] {
            quest1 = &q
        }
        if q.is_unlocked && q.type == current_pair[1] {
            quest2 = &q
        }
    }

    return quest1, quest2
}

//
// :upgrades
Upgrades_UI_Config :: struct {
    menu: struct {
        pos_x: f32,
        pos_y: f32,
        size_x: f32,
        size_y: f32,
        background_sprite: Image_Id,
    },
    button: struct {
        pos_x: f32,
        pos_y: f32,
        size_x: f32,
        size_y: f32,
        bound_x: f32,
        bound_y: f32,
        sprite: Image_Id,
    },
    list: struct {
        start_x: f32,
        start_y: f32,
        item_height: f32,
        spacing: f32,
        title_offset_x: f32,
        button_offset_x: f32,
        button_width: f32,
        button_height: f32,
        bound_x: f32,
        bound_y: f32,
    },
}

Upgrade_Type :: enum {
    dummy_training,
    multiple_dummies,
    auto_spawn,
}

Upgrade_Entry :: struct {
    name: string,
    type: Upgrade_Type,
    current_tier: ^int,
    max_tier: int,
    tier_costs: []int,
    unlocked: bool,
}

render_upgrades_menu_button :: proc() {
    cfg := gs.ui_config.upgrades.button
    button_pos := v2{cfg.pos_x, cfg.pos_y}
    button_size := v2{cfg.size_x, cfg.size_y} * gs.ui.upgrades_button_scale

    draw_sprite_with_size(
        button_pos,
        button_size,
        cfg.sprite,
        pivot = .center_center,
        xform = xform_scale(v2{gs.ui.upgrades_button_scale, gs.ui.upgrades_button_scale}),
        z_layer = .ui,
    )
}

render_upgrades_menu :: proc() {
    if !gs.upgrades_menu_open do return

    cfg := gs.ui_config.upgrades
    push_z_layer(.ui)

    menu_pos := v2{cfg.menu.pos_x, cfg.menu.pos_y}
    menu_size := v2{cfg.menu.size_x, cfg.menu.size_y}
    draw_sprite_with_size(
        menu_pos,
        menu_size,
        cfg.menu.background_sprite,
        pivot = .center_center,
        z_layer = .ui,
    )

    upgrades := [dynamic]Upgrade_Entry{
        {
            name = "Dummy Training",
            type = .dummy_training,
            current_tier = &gs.upgrades_system.current_tier,
            max_tier = len(gs.upgrades_system.dummy_tiers) - 1,
            tier_costs = []int{0, 1000, 5000, 25000},
            unlocked = true,
        },
        {
            name = "Multiple Dummies",
            type = .multiple_dummies,
            current_tier = &gs.upgrades_system.current_multiple_tier,
            max_tier = len(gs.upgrades_system.multiple_dummy_tiers) - 1,
            tier_costs = []int{0, 2000, 10000},
            unlocked = true,
        },
        {
            name = "Auto Spawn",
            type = .auto_spawn,
            current_tier = cast(^int)&gs.upgrades_system.auto_spawn_unlocked,
            max_tier = 1,
            tier_costs = []int{0, 10000},
            unlocked = true,
        },
    }
    defer delete(upgrades)

    list_cfg := cfg.list
    current_y := list_cfg.start_y

    for upgrade in upgrades {
        if !upgrade.unlocked do continue

        name_pos := v2{list_cfg.start_x + list_cfg.title_offset_x, current_y}
        draw_text(
            name_pos,
            upgrade.name,
            scale = 1.2,
            pivot = .center_left,
            z_layer = .ui,
        )

        current := upgrade.current_tier^
        next_tier := current + 1

        if next_tier <= upgrade.max_tier {
            button_pos := v2{
                list_cfg.start_x + list_cfg.button_offset_x,
                current_y,
            }
            button_size := v2{list_cfg.button_width, list_cfg.button_height}
            button_bounds := v2{list_cfg.bound_x, list_cfg.bound_y}
            can_afford := gs.skills_system.gold >= upgrade.tier_costs[next_tier]
            hover := is_point_in_rect(mouse_pos_in_world_space(), button_pos, button_bounds)

            draw_sprite_with_size(
                button_pos,
                button_size,
                img_id = !hover ? .next_skill_button_bg : .next_skill_button_active_bg,
                pivot = .center_center,
                color_override = can_afford ? v4{0,0,0,0} : v4{1,0.5,0.5,0},
                z_layer = .ui,
            )

            draw_text(
                button_pos,
                fmt.tprintf("%d Gold", upgrade.tier_costs[next_tier]),
                scale = 0.9,
                pivot = .center_center,
                z_layer = .ui,
            )

        if hover && can_afford && key_just_pressed(.LEFT_MOUSE) {
                if upgrade.name == "Dummy Training" {
                    unlock_dummy_tier(next_tier)
                } else if upgrade.name == "Multiple Dummies" {
                    unlock_multiple_dummy_tier(next_tier)
                } else if upgrade.name == "Auto Spawn" {
                    gs.upgrades_system.auto_spawn_unlocked = true
                    gs.skills_system.gold -= upgrade.tier_costs[1]

                    add_floating_text_params(
                        v2{-200, 150},
                        "Auto Spawn Unlocked!",
                        v4{1, 0.8, 0, 1},
                        scale = 0.8,
                        target_scale = 1.0,
                        lifetime = 1.0,
                        velocity = v2{0, 75},
                    )
                }
            }
        } else {
            draw_text(
                v2{list_cfg.start_x + list_cfg.button_offset_x, current_y},
                "Max Level",
                scale = 0.9,
                pivot = .center_center,
                z_layer = .ui,
            )
        }

        current_y -= list_cfg.item_height + list_cfg.spacing
    }
}

unlock_dummy_tier :: proc(tier: int) {
    if tier <= 0 || tier >= len(gs.upgrades_system.dummy_tiers) do return

    tier_data := &gs.upgrades_system.dummy_tiers[tier]
    if tier_data.unlocked do return

    if gs.skills_system.gold >= tier_data.cost {
        gs.skills_system.gold -= tier_data.cost
        tier_data.unlocked = true
        gs.upgrades_system.current_tier = tier

        add_floating_text_params(
            v2{-200, 150},
            fmt.tprintf("Unlocked Training Tier %d!", tier),
            v4{1, 0.8, 0, 1},
            scale = 0.8,
            target_scale = 1.0,
            lifetime = 1.0,
            velocity = v2{0, 75},
        )
    }
}

unlock_multiple_dummy_tier :: proc(tier: int) {
    if tier <= 0 || tier >= len(gs.upgrades_system.multiple_dummy_tiers) do return

    tier_data := &gs.upgrades_system.multiple_dummy_tiers[tier]
    if tier_data.unlocked do return

    if gs.skills_system.gold >= tier_data.cost {
        gs.skills_system.gold -= tier_data.cost
        tier_data.unlocked = true
        gs.upgrades_system.current_multiple_tier = tier

        add_floating_text_params(
            v2{-200, 150},
            fmt.tprintf("Unlocked Multiple Dummies Tier %d!", tier),
            v4{1, 0.8, 0, 1},
            scale = 0.8,
            target_scale = 1.0,
            lifetime = 1.0,
            velocity = v2{0, 75},
        )
    }
}

Upgrades_System :: struct {
    dummy_tiers: [4]Dummy_Tier,
    current_tier: int,
    multiple_dummy_tiers: [3]Multiple_Dummy_Tier,
    current_multiple_tier: int,
    auto_spawn_unlocked: bool,
    frames_since_last_spawn: f32,
}

Dummy_Tier :: struct {
    level: int,
    health: f32,
    xp_multiplier: f32,
    cost: int,
    frames: []Image_Id,
    base_sprite: Image_Id,
    unlocked: bool,
}

Multiple_Dummy_Tier :: struct {
    unlocked: bool,
    cost: int,
    max_dummies: int,
}

init_upgrades_system :: proc() -> Upgrades_System {
    system := Upgrades_System{
        current_tier = 0,
        current_multiple_tier = 0,
        auto_spawn_unlocked = false,
        frames_since_last_spawn = 0,
    }

    dummy_training_grounds_init(&system)
    multiple_dummy_tier_init(&system)

    return system
}

dummy_training_grounds_init :: proc(system: ^Upgrades_System) {
    tier0_frames := make([]Image_Id, 6)
    tier0_frames[0] = .dummy_hit1
    tier0_frames[1] = .dummy_hit2
    tier0_frames[2] = .dummy_hit3
    tier0_frames[3] = .dummy_hit4
    tier0_frames[4] = .dummy_hit5
    tier0_frames[5] = .dummy_hit1

    system.dummy_tiers[0] = Dummy_Tier {
        level = 0,
        health = 100,
        xp_multiplier = 1.0,
        cost = 0,
        frames = tier0_frames,
        base_sprite = .dummy_hit1,
        unlocked = true,
    }

    tier1_frames := make([]Image_Id, 6)
    tier1_frames[0] = .dummy1_hit1
    tier1_frames[1] = .dummy1_hit2
    tier1_frames[2] = .dummy1_hit3
    tier1_frames[3] = .dummy1_hit4
    tier1_frames[4] = .dummy1_hit5
    tier1_frames[5] = .dummy1_hit1

    system.dummy_tiers[1] = Dummy_Tier{
        level = 1,
        health = 250,
        xp_multiplier = 2.5,
        cost = 1000,
        frames = tier1_frames,
        base_sprite = .dummy1_hit1,
        unlocked = false,
    }

    tier2_frames := make([]Image_Id, 6)
    tier2_frames[0] = .dummy2_hit1
    tier2_frames[1] = .dummy2_hit2
    tier2_frames[2] = .dummy2_hit3
    tier2_frames[3] = .dummy2_hit4
    tier2_frames[4] = .dummy2_hit5
    tier2_frames[5] = .dummy2_hit1

    system.dummy_tiers[2] = Dummy_Tier{
        level = 2,
        health = 600,
        xp_multiplier = 6.0,
        cost = 5000,
        frames = tier2_frames,
        base_sprite = .dummy2_hit1,
        unlocked = false,
    }

    tier3_frames := make([]Image_Id, 6)
    tier3_frames[0] = .dummy3_hit1
    tier3_frames[1] = .dummy3_hit2
    tier3_frames[2] = .dummy3_hit3
    tier3_frames[3] = .dummy3_hit4
    tier3_frames[4] = .dummy3_hit5
    tier3_frames[5] = .dummy3_hit1

    system.dummy_tiers[3] = Dummy_Tier{
        level = 3,
        health = 1500,
        xp_multiplier = 15.0,
        cost = 25000,
        frames = tier3_frames,
        base_sprite = .dummy3_hit1,
        unlocked = false,
    }
}

multiple_dummy_tier_init :: proc(system: ^Upgrades_System) {
    system.multiple_dummy_tiers[0] = Multiple_Dummy_Tier{
        unlocked = true,
        cost = 0,
        max_dummies = 1,
    }
    system.multiple_dummy_tiers[1] = Multiple_Dummy_Tier{
        unlocked = false,
        cost = 2000,
        max_dummies = 2,
    }
    system.multiple_dummy_tiers[2] = Multiple_Dummy_Tier{
        unlocked = false,
        cost = 10000,
        max_dummies = 4,
    }
}

//
// :skills
FOCUS_MODE_DURATION :: 5.0
FOCUS_MODE_BUTTON_DURATION :: 3.0
FOCUS_MODE_XP_MULTIPLIER :: 2.0
PASSIVE_XP_INTERVAL :: 2.5
PASSIVE_XP_BASE_RATE :: 0.005

Skill_Type :: enum {
    nil,
    xp_boost,
    strength_boost,
    speed_boost,
    critical_boost,
    storm_arrows,
    mystic_fletcher,
    warrior_stamina,
    formation_mastery,
    battle_meditation,
    war_preparation,
    tactical_analysis,
    commanders_authority,
}

Skill :: struct {
    type: Skill_Type,
    name: string,
    level: int,
    current_xp: int,
    xp_to_next_level: int,
    description: string,
    is_unlocked: bool,
    display_xp: f32,
}

Skills_Menu_Type :: enum {
    normal,
    advanced,
}

Skills_System :: struct {
    skills: [dynamic]Skill,
    advanced_skills: [dynamic]Skill,
    active_skill: ^Skill,
    dummies_killed: int,
    is_unlocked: bool,
    gold: int,
    menu_open: bool,
    active_menu: Skills_Menu_Type,
    passive_xp_timer: f32,
    focus_mode_active: bool,
    focus_mode_timer: f32,
    focus_mode_button_visible: bool,
    focus_mode_button_timer: f32,
}

Skills_UI_Config :: struct {
    menu: struct {
        pos_x: f32,
        pos_y: f32,
        size_x: f32,
        size_y: f32,
        background_sprite: Image_Id,
    },
    button: struct {
        pos_y: f32,
        size_x: f32,
        size_y: f32,
        sprite: Image_Id,
    },
    switch_button: struct {
        offset_x: f32,
        offset_y: f32,
        size_x: f32,
        size_y: f32,
        bounds_x: f32,
        bounds_y: f32,
        text_scale: f32,
        sprite: Image_Id,
    },
    focus_mode: struct{
        button_pos_x: f32,
        button_pos_y: f32,
        button_size_x: f32,
        button_size_y: f32,
        button_bounds_x: f32,
        button_bounds_y: f32,
        glow_size_x: f32,
        glow_size_y: f32,
        back_sprite: Image_Id,
        front_sprite: Image_Id,
        text_scale: f32,
        text_pos_x: f32,
        text_pos_y: f32,
        focus_text_pos_x: f32,
        focus_text_pos_y: f32,
    },
    next_skill: struct {
        offset_x: f32,
        offset_y: f32,
        empty_panel_pos_x: f32,
        empty_panel_pos_y: f32,
        empty_panel_size_x: f32,
        empty_panel_size_y: f32,
        empty_title_offset_x: f32,
        empty_title_offset_y: f32,
        panel_size_x: f32,
        panel_size_y: f32,
        title_offset_x: f32,
        title_offset_y: f32,
        name_offset_x: f32,
        name_offset_y: f32,
        cost_offset_x: f32,
        cost_offset_y: f32,
        button_offset_x: f32,
        button_offset_y: f32,
        button_size_x: f32,
        button_size_y: f32,
        button_bnds_x: f32,
        button_bnds_y: f32,
    },
    unlocked_skills: struct {
        start_offset_x: f32,
        start_offset_y: f32,
        spacing_y: f32,
        radio_button: struct {
            offset_x: f32,
            offset_y: f32,
            size_x: f32,
            size_y: f32,
            bounds_x: f32,
            bounds_y: f32,
            selected_sprite: Image_Id,
            unselected_sprite: Image_Id,
        },
        skill_name: struct {
            offset_x: f32,
            offset_y: f32,
            scale: f32,
        },
        level_text: struct {
            offset_x: f32,
            offset_y: f32,
            scale: f32,
        },
        xp_bar: struct {
            bar_width: f32,
            bar_height: f32,
            bar_pos_x: f32,
            bar_pos_y: f32,
            bar_sprite: Image_Id,
            zlayer_xp: ZLayer,
            zlayer_xp_2: ZLayer,
            fill_speed: f32,
            rect_width: f32,
            rect_height: f32,
            rect_pos_x: f32,
            rect_pos_y: f32,
        }
    },
    tooltip: struct {
        offset_x: f32,
        offset_y: f32,
        size_x: f32,
        size_y: f32,
        padding_x: f32,
        padding_y: f32,
        text_pos_x: f32,
        text_pos_y: f32,
        line_spacing: f32,
        sprite: Image_Id,
    },
}

init_skills_system :: proc() -> Skills_System {
    system := Skills_System{
        skills = make([dynamic]Skill),
        advanced_skills = make([dynamic]Skill),
        active_skill = nil,
        dummies_killed = 0,
        is_unlocked = false,
        gold = 100000,
        menu_open = false,
        active_menu = .normal,
        passive_xp_timer = 0,
    }

    skill_data := []struct{type: Skill_Type, name, desc: string}{
        {.xp_boost, "Experience Mastery", "Increases experience gained from destroying dummies by 5% per level"},
        {.strength_boost, "Strength Mastery", "Increases arrow damage by 5% per level"},
        {.speed_boost, "Speed Mastery", "Increases attack speed by 5% per level"},
        {.critical_boost, "Critical Mastery", "Increases critical hit chance by 1% per level"},
        {.storm_arrows, "Storm of Arrows", "Gain 5% chance per level to fire an additional arrow"},
        {.mystic_fletcher, "Mystic Fletcher", "Gain 5% chance per level to fire an additional elemental arrow that deals extra damage"},
        {.warrior_stamina, "Warrior's Stamina", "Gain 5% change per leve to instantly fire another arrow after a critical hit"}
    }

    advanced_skill_data := []struct{type: Skill_Type, name, desc: string} {
        {.formation_mastery, "Formation Mastery", "When a dummy dies, gain a 2% chance per level to spawn a ghost dummy that lasts for 5 seconds and grants double XP when killed"},
        {.battle_meditation, "Battle Meditation", "Gain a 1% chance per level to trigger Focus Mode opportunity, which greatly increases XP gains for a short duration"},
        {.war_preparation, "War Preparation", "Increases gold gained from all sources by 2% per level"},
        {.tactical_analysis, "Tactical Analysis", "Increases chance for dummies to spawn with weak points that deal bonus damage when hit"},
        {.commanders_authority, "Commander's Authority", "All unlocked skills gain passive XP over time, increasing by 0.5% per level"},
    }

    for data in skill_data {
        skill := Skill {
            type = data.type,
            name = data.name,
            level = 0,
            current_xp = 0,
            xp_to_next_level = 100,
            description = data.desc,
            is_unlocked = false,
        }

        append(&system.skills, skill)
    }

    for data in advanced_skill_data {
        skill := Skill {
            type = data.type,
            name = data.name,
            level = 0,
            current_xp = 0,
            xp_to_next_level = 100,
            description = data.desc,
            is_unlocked = false,
        }

        append(&system.advanced_skills, skill)
    }

    return system
}

calculate_xp_boost :: proc(system: ^Skills_System) -> f32 {
    for &skill in system.skills {
        if skill.is_unlocked && skill.type == .xp_boost {
            base_boost := XP.BASE_XP_BOOST
            level_boost := XP.XP_BOOST_PER_LEVEL * f32(skill.level - 1)
            return 1.0 + base_boost + level_boost
        }
    }
    return 1.0
}

calculate_passive_xp_for_skill :: proc(commander_level: int, xp_to_next: int) -> int {
    percentage := PASSIVE_XP_BASE_RATE * f32(commander_level)
    xp_gain := int(f32(xp_to_next) * percentage)

    return max(1, xp_gain)
}

update_passive_skill_xp :: proc(dt: f32) {
    if system := &gs.skills_system; system.is_unlocked {
        system.passive_xp_timer -= dt
        if system.passive_xp_timer <= 0 {
            system.passive_xp_timer = PASSIVE_XP_INTERVAL

            commander_level := 0
            for skill in system.advanced_skills {
                if skill.is_unlocked && skill.type == .commanders_authority {
                    commander_level = skill.level
                    break
                }
            }

            if commander_level > 0 {
                for &skill in system.skills {
                    if skill.is_unlocked && skill.level < XP.MAX_LEVEL {
                        passive_xp := calculate_passive_xp_for_skill(commander_level, skill.xp_to_next_level)
                        if passive_xp > 0 {
                            prev_level := skill.level
                            skill.current_xp += passive_xp

                            for skill.current_xp >= skill.xp_to_next_level {
                                skill.current_xp -= skill.xp_to_next_level
                                skill.level += 1

                                if skill.level >= XP.MAX_LEVEL {
                                    skill.level = XP.MAX_LEVEL
                                    skill.current_xp = 0
                                    break
                                }

                                skill.xp_to_next_level = int(f32(skill.xp_to_next_level) * XP.LEVEL_XP_MULTIPLIER)
                            }

                            if skill.level > prev_level {
                                check_quest_unlocks(&skill)
                                add_floating_text_params(
                                    v2{-200, 150},
                                    fmt.tprintf("%s reached level %d!", skill.name, skill.level),
                                    v4{0.5, 0.5, 1, 1},
                                    scale = 0.8,
                                    target_scale = 1.0,
                                    lifetime = 1.0,
                                    velocity = v2{0, 50},
                                )
                            }
                        }
                    }
                }

                for &skill in system.advanced_skills {
                    if skill.is_unlocked && skill.level < XP.MAX_LEVEL {
                        passive_xp := calculate_passive_xp_for_skill(commander_level, skill.xp_to_next_level)
                        if passive_xp > 0 {
                            prev_level := skill.level
                            skill.current_xp += passive_xp

                            for skill.current_xp >= skill.xp_to_next_level {
                                skill.current_xp -= skill.xp_to_next_level
                                skill.level += 1

                                if skill.level >= XP.MAX_LEVEL {
                                    skill.level = XP.MAX_LEVEL
                                    skill.current_xp = 0
                                    break
                                }

                                skill.xp_to_next_level = int(f32(skill.xp_to_next_level) * XP.LEVEL_XP_MULTIPLIER)
                            }

                            if skill.level > prev_level {
                                check_quest_unlocks(&skill)
                                add_floating_text_params(
                                    v2{-200, 150},
                                    fmt.tprintf("%s reached level %d!", skill.name, skill.level),
                                    v4{0.5, 0.5, 1, 1},
                                    scale = 0.8,
                                    target_scale = 1.0,
                                    lifetime = 1.0,
                                    velocity = v2{0, 50},
                                )
                            }
                        }
                    }
                }

                total_skills := 0
                for skill in system.skills do if skill.is_unlocked do total_skills += 1
                for skill in system.advanced_skills do if skill.is_unlocked do total_skills += 1

                if total_skills > 0 {
                    example_xp := calculate_passive_xp_for_skill(commander_level, 100) // Using base XP requirement for display
                    add_floating_text_params(
                        v2{-200, 100},
                        fmt.tprintf("Passive XP: +%.1f%% to all skills", PASSIVE_XP_BASE_RATE * f32(commander_level) * 100),
                        v4{0.5, 0.5, 1, 0.8},
                        scale = 0.7,
                        target_scale = 0.8,
                        lifetime = 0.8,
                        velocity = v2{0, 40},
                    )
                }
            }
        }
    }
}

calculate_skill_bonus :: proc(skill: ^Skill) -> f32 {
    if skill == nil {
        return 0.0
    }

    base_bonus: f32
    #partial switch skill.type {
        // NORMAL
        case .xp_boost: base_bonus = 0.05 // 0.05
        case .strength_boost: base_bonus = 0.05 // 0.05
        case .speed_boost: base_bonus = 0.05 // 0.05
        case .critical_boost: base_bonus = 0.01 // 0.01
        case .storm_arrows: base_bonus = 0.05 // 0.05
        case .mystic_fletcher: base_bonus = 0.05 // 0.05
        case .warrior_stamina: base_bonus = 0.02 // 0.02

        // ADVANCED
        case .formation_mastery: base_bonus = 0.02 // 0.02
        case .battle_meditation: base_bonus = 0.01 // 0.01
        case .war_preparation: base_bonus = 0.02 // 0.02
        case .tactical_analysis: base_bonus = 1.0 // 0.02
        case .commanders_authority: base_bonus = PASSIVE_XP_BASE_RATE
        case: return 0.0
    }

    return base_bonus * f32(skill.level)
}

add_xp_to_active_skill :: proc(system: ^Skills_System, base_xp: int) -> int {
    if system.active_skill == nil {
        return 0
    }

    if system.active_skill.level >= XP.MAX_LEVEL {
        return 0
    }

    xp_multiplier := calculate_xp_boost(system)
    if system.focus_mode_active {
        xp_multiplier *= FOCUS_MODE_XP_MULTIPLIER
    }

    tier_multiplier := gs.upgrades_system.dummy_tiers[gs.upgrades_system.current_tier].xp_multiplier
    total_xp := int(f32(base_xp) * xp_multiplier * tier_multiplier)
    prev_level := system.active_skill.level

    system.active_skill.current_xp += total_xp

    for system.active_skill.current_xp >= system.active_skill.xp_to_next_level {
        system.active_skill.current_xp -= system.active_skill.xp_to_next_level
        system.active_skill.level += 1

        if system.active_skill.level >= XP.MAX_LEVEL {
            system.active_skill.level = XP.MAX_LEVEL
            system.active_skill.current_xp = 0
            break
        }

        system.active_skill.xp_to_next_level = int(f32(system.active_skill.xp_to_next_level) * XP.LEVEL_XP_MULTIPLIER)
    }

    if system.active_skill.level > prev_level {
        check_quest_unlocks(system.active_skill)
    }

    return total_xp
}

check_skills_unlock :: proc(system: ^Skills_System) {
    if system.is_unlocked {
        return
    }

    if system.dummies_killed >= 3 {
        system.is_unlocked = true
        system.skills[0].is_unlocked = true
        system.active_skill = &system.skills[0]
    }
}

get_skill_cost :: proc(type: Skill_Type) -> int {
    switch type {
        case .nil: return 0
        case .xp_boost: return 0
        case .strength_boost: return 30
        case .speed_boost: return 300
        case .critical_boost: return 1100
        case .storm_arrows: return 9000
        case .mystic_fletcher: return 50000
        case .warrior_stamina: return 75000
        case .formation_mastery: return 100000
        case .battle_meditation: return 130000
        case .war_preparation: return 160000
        case .tactical_analysis: return 180000
        case .commanders_authority: return 200000
    }
    return 0
}

render_skills_ui :: proc() {
    if !gs.skills_system.menu_open do return
    if gs.ui.skills_menu_alpha <= 0 do return

    cfg := gs.ui_config.skills
    alpha := gs.ui.skills_menu_alpha
    push_z_layer(.ui)

    menu_pos := v2{cfg.menu.pos_x, cfg.menu.pos_y}
    menu_size := v2{cfg.menu.size_x, cfg.menu.size_y}
    draw_sprite_with_size(
        menu_pos,
        menu_size,
        cfg.menu.background_sprite,
        pivot = .center_center,
        color_override = v4{0,0,0,1-alpha})

    button_pos := menu_pos + v2{cfg.switch_button.offset_x, cfg.switch_button.offset_y}
    button_size := v2{cfg.switch_button.size_x, cfg.switch_button.size_y}
    btn_bounds := v2{cfg.switch_button.bounds_x, cfg.switch_button.bounds_y}
    hover := is_point_in_rect(mouse_pos_in_world_space(), button_pos, btn_bounds)

    draw_sprite_with_size(
        button_pos,
        button_size,
        img_id = !hover ? Image_Id.next_skill_button_bg : Image_Id.next_skill_button_active_bg,
        pivot = .center_center,
        z_layer = .ui,
    )

    button_text := gs.skills_system.active_menu == .normal ? "Advanced" : "Normal"
    draw_text(
        button_pos,
        button_text,
        scale = auto_cast  cfg.switch_button.text_scale,
        pivot = .center_center,
        z_layer = .ui,
    )

    if hover && key_just_pressed(.LEFT_MOUSE) {
        gs.skills_system.active_menu = gs.skills_system.active_menu == .normal ? .advanced : .normal
    }

    if gs.skills_system.active_menu == .normal {
        draw_normal_skills_content(menu_pos, alpha)
    }else {
        draw_advanced_skills_content(menu_pos, alpha)
    }
}

draw_normal_skills_content :: proc(menu_pos: Vector2, alpha: f32){
    cfg := gs.ui_config.skills

    next_skill_pos := menu_pos + v2{cfg.next_skill.offset_x, cfg.next_skill.offset_y}

    next_skill: ^Skill
    for &skill in gs.skills_system.skills {
        if !skill.is_unlocked {
            next_skill = &skill
            break
        }
    }

    if next_skill != nil{
        draw_next_skill_panel(next_skill, next_skill_pos, alpha)
    }else {
        draw_empty_next_skill_panel()
    }

    unlocked_pos := menu_pos + v2{cfg.unlocked_skills.start_offset_x, cfg.unlocked_skills.start_offset_y}
    draw_unlocked_normal_skills(unlocked_pos, alpha)
}

draw_advanced_skills_content :: proc(menu_pos: Vector2, alpha: f32){
    cfg := gs.ui_config.skills

    next_skill_pos := menu_pos + v2{cfg.next_skill.offset_x, cfg.next_skill.offset_y}

    next_skill: ^Skill
    for &skill in gs.skills_system.advanced_skills {
        if !skill.is_unlocked {
            next_skill = &skill
            break
        }
    }

    if next_skill != nil {
        draw_next_skill_panel(next_skill, next_skill_pos, alpha)
    }

    unlocked_pos := menu_pos + v2{cfg.unlocked_skills.start_offset_x, cfg.unlocked_skills.start_offset_y}
    draw_unlocked_advanced_skills(unlocked_pos, alpha)
}

draw_unlocked_normal_skills :: proc(start_pos: Vector2, alpha: f32) {
    cfg := gs.ui_config.skills.unlocked_skills
    menu_cfg := gs.ui_config.skills.menu
    push_z_layer(.ui)

    content_height := menu_cfg.size_y * 0.7
    content_width := menu_cfg.size_x * 0.8

    spacing := cfg.spacing_y
    visible_height := content_height

    unlocked_count := 0
    total_content_height := 0.0
    for skill in gs.skills_system.skills {
        if skill.is_unlocked {
            unlocked_count += 1
            total_content_height += f64(spacing)
        }
    }

    if unlocked_count == 0 do return

    if !gs.ui.skills_scroll_initialized {
        gs.ui.skills_scroll_pos = 0
        gs.ui.skills_scroll_initialized = true
    }

    if content_bounds := aabb_make(start_pos, v2{content_width, content_height}, Pivot.center_center);
       aabb_contains(content_bounds, mouse_pos_in_world_space()) {
        scroll_speed := 40.0
        if key_down(.UP) {
            gs.ui.skills_scroll_pos = max(0, gs.ui.skills_scroll_pos - f32(scroll_speed) * f32(sapp.frame_duration()))
        }
        if key_down(.DOWN) {
            max_scroll := max(0, total_content_height - f64(visible_height))
            gs.ui.skills_scroll_pos = min(f32(max_scroll), gs.ui.skills_scroll_pos + f32(scroll_speed) * f32(sapp.frame_duration()))
        }
    }

    content_top := start_pos.y + content_height * 0.5
    content_bottom := start_pos.y - content_height * 0.5

    pos := v2{start_pos.x, content_top - spacing * 0.5} - v2{0, gs.ui.skills_scroll_pos}

    mouse_pos := mouse_pos_in_world_space()

    for &skill, i in gs.skills_system.skills {
        if !skill.is_unlocked do continue
        if pos.y + spacing < content_bottom || pos.y > content_top do continue

        bar_width := cfg.xp_bar.bar_width
        bar_height := cfg.xp_bar.bar_height
        rect_width := cfg.xp_bar.rect_width
        rect_height := cfg.xp_bar.rect_height
        bar_pos := pos + v2{cfg.xp_bar.bar_pos_x, cfg.xp_bar.bar_pos_y}
        rect_pos := pos + v2{cfg.xp_bar.rect_pos_x, cfg.xp_bar.rect_pos_y}

        draw_sprite_with_size(
            bar_pos,
            v2{bar_width, bar_height},
            cfg.xp_bar.bar_sprite,
            pivot = .bottom_center,
            z_layer = cfg.xp_bar.zlayer_xp
        )

        target_xp_ratio := f32(skill.current_xp) / f32(skill.xp_to_next_level)
        animate_to_target_f32(&skill.display_xp, target_xp_ratio, f32(sapp.frame_duration()) * cfg.xp_bar.fill_speed)

        draw_rect_aabb(
            rect_pos,
            v2{rect_width * skill.display_xp, rect_height},
            col = Colors.xp_bar_fill,
            z_layer = cfg.xp_bar.zlayer_xp_2
        )

        is_active := gs.skills_system.active_skill == &gs.skills_system.skills[i]

        if pos.y <= content_top && pos.y >= content_bottom {
            radio_pos := pos + v2{cfg.radio_button.offset_x, cfg.radio_button.offset_y}
            radio_sprite := is_active ? cfg.radio_button.selected_sprite : cfg.radio_button.unselected_sprite
            radio_size := v2{cfg.radio_button.size_x, cfg.radio_button.size_y}
            fitted_size := fit_size_to_square(radio_size)
            radio_bounds := v2{cfg.radio_button.bounds_x, cfg.radio_button.bounds_y}

            draw_nores_sprite_with_size(
                radio_pos,
                fitted_size,
                radio_sprite,
                pivot = .center_center,
                color_override = v4{1,1,1,0},
                z_layer = .ui
            )

            debug_button := false
            click_area := aabb_make(radio_pos, radio_size + radio_bounds, Pivot.center_center)

            if debug_button {
                draw_rect_aabb_actually(click_area, col=v4{1,0,0,0.2}, z_layer=.ui)
            }

            name_pos := pos + v2{cfg.skill_name.offset_x, cfg.skill_name.offset_y}
            draw_text(
                name_pos,
                skill.name,
                col = Colors.text * v4{1,1,1,alpha},
                scale = auto_cast cfg.skill_name.scale,
                pivot = .center_left,
                z_layer = .ui
            )

            level_pos := pos + v2{cfg.level_text.offset_x, cfg.level_text.offset_y}
            draw_text(
                level_pos,
                fmt.tprintf("Level %d", skill.level),
                col = Colors.text * v4{1,1,1,alpha},
                scale = auto_cast cfg.level_text.scale,
                pivot = .center_left,
                z_layer = .ui
            )

            mouse_over_radio := aabb_contains(click_area, mouse_pos)

            if mouse_over_radio {
                gs.ui.skills_hover_tooltip_active = true
                gs.ui.skills_hover_tooltip_skill = &gs.skills_system.skills[i]

                tooltip_cfg := gs.ui_config.skills.tooltip
                tooltip_pos := pos + v2{tooltip_cfg.offset_x, tooltip_cfg.offset_y}
                draw_skill_tooltip(&gs.skills_system.skills[i])

                if key_just_pressed(.LEFT_MOUSE) {
                    old_active := gs.skills_system.active_skill
                    gs.skills_system.active_skill = &gs.skills_system.skills[i]
                }
            }
        }

        pos.y -= spacing
    }
}

draw_unlocked_advanced_skills :: proc(start_pos: Vector2, alpha: f32) {
    cfg := gs.ui_config.skills.unlocked_skills
    menu_cfg := gs.ui_config.skills.menu
    push_z_layer(.ui)

    content_height := menu_cfg.size_y * 0.7
    content_width := menu_cfg.size_x * 0.8

    spacing := cfg.spacing_y
    visible_height := content_height

    unlocked_count := 0
    total_content_height := 0.0
    for skill in gs.skills_system.advanced_skills {
        if skill.is_unlocked {
            unlocked_count += 1
            total_content_height += f64(spacing)
        }
    }

    if unlocked_count == 0 do return

    if !gs.ui.skills_scroll_initialized {
        gs.ui.skills_scroll_pos = 0
        gs.ui.skills_scroll_initialized = true
    }

    if content_bounds := aabb_make(start_pos, v2{content_width, content_height}, Pivot.center_center);
       aabb_contains(content_bounds, mouse_pos_in_world_space()) {
        scroll_speed := 40.0
        if key_down(.UP) {
            gs.ui.skills_scroll_pos = max(0, gs.ui.skills_scroll_pos - f32(scroll_speed) * f32(sapp.frame_duration()))
        }
        if key_down(.DOWN) {
            max_scroll := max(0, total_content_height - f64(visible_height))
            gs.ui.skills_scroll_pos = min(f32(max_scroll), gs.ui.skills_scroll_pos + f32(scroll_speed) * f32(sapp.frame_duration()))
        }
    }

    content_top := start_pos.y + content_height * 0.5
    content_bottom := start_pos.y - content_height * 0.5

    pos := v2{start_pos.x, content_top - spacing * 0.5} - v2{0, gs.ui.skills_scroll_pos}

    mouse_pos := mouse_pos_in_world_space()

    for &skill, i in gs.skills_system.advanced_skills {
        if !skill.is_unlocked do continue
        if pos.y + spacing < content_bottom || pos.y > content_top do continue

        bar_width := cfg.xp_bar.bar_width
        bar_height := cfg.xp_bar.bar_height
        rect_width := cfg.xp_bar.rect_width
        rect_height := cfg.xp_bar.rect_height
        bar_pos := pos + v2{cfg.xp_bar.bar_pos_x, cfg.xp_bar.bar_pos_y}
        rect_pos := pos + v2{cfg.xp_bar.rect_pos_x, cfg.xp_bar.rect_pos_y}

        draw_sprite_with_size(
            bar_pos,
            v2{bar_width, bar_height},
            cfg.xp_bar.bar_sprite,
            pivot = .bottom_center,
            z_layer = cfg.xp_bar.zlayer_xp
        )

        target_xp_ratio := f32(skill.current_xp) / f32(skill.xp_to_next_level)
        animate_to_target_f32(&skill.display_xp, target_xp_ratio, f32(sapp.frame_duration()) * cfg.xp_bar.fill_speed)

        draw_rect_aabb(
            rect_pos,
            v2{rect_width * skill.display_xp, rect_height},
            col = Colors.xp_bar_fill,
            z_layer = cfg.xp_bar.zlayer_xp_2
        )

        xp_text_pos := bar_pos + v2{bar_width + 10, 0}
        draw_text(
            xp_text_pos,
            fmt.tprintf("%d/%d", skill.current_xp, skill.xp_to_next_level),
            col = Colors.text * v4{1,1,1,1},
            scale = 0.8,
            pivot = .center_left,
            z_layer = cfg.xp_bar.zlayer_xp
        )

        is_active := gs.skills_system.active_skill == &gs.skills_system.advanced_skills[i]

        if pos.y <= content_top && pos.y >= content_bottom {
            radio_pos := pos + v2{cfg.radio_button.offset_x, cfg.radio_button.offset_y}
            radio_sprite := is_active ? cfg.radio_button.selected_sprite : cfg.radio_button.unselected_sprite
            radio_size := v2{cfg.radio_button.size_x, cfg.radio_button.size_y}
            fitted_size := fit_size_to_square(radio_size)
            radio_bounds := v2{cfg.radio_button.bounds_x, cfg.radio_button.bounds_y}

            draw_nores_sprite_with_size(
                radio_pos,
                fitted_size,
                radio_sprite,
                pivot = .center_center,
                color_override = v4{1,1,1,0},
                z_layer = .ui
            )

            debug_button := false
            click_area := aabb_make(radio_pos, radio_size + radio_bounds, Pivot.center_center)

            if debug_button {
                draw_rect_aabb_actually(click_area, col=v4{1,0,0,0.2}, z_layer=.ui)
            }

            name_pos := pos + v2{cfg.skill_name.offset_x, cfg.skill_name.offset_y}
            draw_text(
                name_pos,
                skill.name,
                col = Colors.text * v4{1,1,1,alpha},
                scale = auto_cast cfg.skill_name.scale,
                pivot = .center_left,
                z_layer = .ui
            )

            level_pos := pos + v2{cfg.level_text.offset_x, cfg.level_text.offset_y}
            draw_text(
                level_pos,
                fmt.tprintf("Level %d", skill.level),
                col = Colors.text * v4{1,1,1,alpha},
                scale = auto_cast cfg.level_text.scale,
                pivot = .center_left,
                z_layer = .ui
            )

            mouse_over_radio := aabb_contains(click_area, mouse_pos)

            if mouse_over_radio {
                gs.ui.skills_hover_tooltip_active = true
                gs.ui.skills_hover_tooltip_skill = &gs.skills_system.advanced_skills[i]

                tooltip_cfg := gs.ui_config.skills.tooltip
                tooltip_pos := pos + v2{tooltip_cfg.offset_x, tooltip_cfg.offset_y}
                draw_skill_tooltip(&gs.skills_system.advanced_skills[i])

                if key_just_pressed(.LEFT_MOUSE) {
                    old_active := gs.skills_system.active_skill
                    gs.skills_system.active_skill = &gs.skills_system.advanced_skills[i]
                }
            }
        }

        pos.y -= spacing
    }
}

unlock_skill :: proc(system: ^Skills_System, skill: ^Skill) {
    cost := get_skill_cost(skill.type)
    if system.gold >= cost {
        system.gold -= cost

        is_advanced := false
        for &s in system.advanced_skills {
            if s.type == skill.type {
                is_advanced = true
                break
            }
        }

        if is_advanced {
            old_skills := make([dynamic]Skill, len(system.advanced_skills))
            defer delete(old_skills)

            copy(old_skills[:], system.advanced_skills[:])
            clear(&system.advanced_skills)

            for s in old_skills {
                if s.is_unlocked {
                    append(&system.advanced_skills, s)
                }
            }

            for &s in old_skills {
                if s.type == skill.type {
                    s.is_unlocked = true
                    append(&system.advanced_skills, s)
                }
            }

            for s in old_skills {
                if !s.is_unlocked && s.type != skill.type {
                    append(&system.advanced_skills, s)
                }
            }

            if system.active_skill == nil {
                system.active_skill = skill
            }
        } else {
            old_skills := make([dynamic]Skill, len(system.skills))
            defer delete(old_skills)

            copy(old_skills[:], system.skills[:])
            clear(&system.skills)

            for s in old_skills {
                if s.is_unlocked {
                    append(&system.skills, s)
                }
            }

            for &s in old_skills {
                if s.type == skill.type {
                    s.is_unlocked = true
                    append(&system.skills, s)
                }
            }

            for s in old_skills {
                if !s.is_unlocked && s.type != skill.type {
                    append(&system.skills, s)
                }
            }

            if system.active_skill == nil {
                system.active_skill = skill
            }
        }
    }
}

find_next_locked_skill :: proc(system: ^Skills_System) -> ^Skill {
    for &skill in system.skills {
        if !skill.is_unlocked {
            return &skill
        }
    }
    return nil
}

any_skill_unlocked :: proc(system: ^Skills_System) -> bool {
    for skill in system.skills {
        if skill.is_unlocked {
            return true
        }
    }
    return false
}

draw_empty_next_skill_panel :: proc() {
    cfg := gs.ui_config.skills.next_skill

    pos := v2{cfg.empty_panel_pos_x, cfg.empty_panel_pos_y}
    panel_size := v2{cfg.empty_panel_size_x, cfg.empty_panel_size_y}
    draw_sprite_with_size(
        pos,
        panel_size,
        .next_skill_panel_bg,
        pivot = .center_center,
        color_override = v4{1,1,1,0},
        z_layer = .ui
    )

    title_pos := pos + v2{cfg.empty_title_offset_x, cfg.empty_title_offset_y}
    draw_text(
        title_pos,
        "All Skills Unlocked!",
        col = Colors.text * v4{1,1,1,1},
        scale = 1.2,
        pivot = .center_left,
        z_layer = .ui,
    )
}

draw_next_skill_panel :: proc(skill: ^Skill, pos: Vector2, alpha: f32) {
    cfg := gs.ui_config.skills.next_skill

    if skill == nil do return

    if skill.is_unlocked {
        next_skill := find_next_locked_skill(&gs.skills_system)
        if next_skill == nil do return
        skill := next_skill
    }

    push_z_layer(.ui)

    panel_size := v2{cfg.panel_size_x, cfg.panel_size_y}
    draw_sprite(pos,
        .next_skill_panel_bg,
        pivot = .center_center,
        color_override = v4{1,1,1,0},
        z_layer = .ui
    )

    cost := get_skill_cost(skill.type)
    can_afford := gs.skills_system.gold >= cost

    title_pos := pos + v2{cfg.title_offset_x, cfg.title_offset_y}
    draw_text(
        title_pos,
        "Next Available Skill",
        col = Colors.text * v4{1,1,1,1},
        scale = 1.2,
        pivot = .center_left,
        z_layer = .ui,
    )

    name_pos := pos + v2{cfg.name_offset_x, cfg.name_offset_y}
    draw_text(
        name_pos,
        skill.name,
        col = Colors.text * v4{1,1,1,1},
        pivot = .center_left,
        z_layer = .ui
    )

    cost_pos := pos + v2{cfg.cost_offset_x, cfg.cost_offset_y}
    draw_text(
        cost_pos,
        fmt.tprintf("Cost: %d gold", cost),
        col = Colors.text * v4{1,1,1,1},
        pivot = .center_left,
        z_layer = .ui
    )

    button_pos := pos + v2{cfg.button_offset_x, cfg.button_offset_y}
    button_size := v2{cfg.button_size_x, cfg.button_size_y}
    bounds_act := v2{cfg.button_bnds_x, cfg.button_bnds_y}

    draw_sprite_with_size(
        button_pos,
        button_size,
        .next_skill_button_bg,
        pivot = .center_center,
        color_override = can_afford ? v4{1,1,1,0} : v4{1,0.5,0.5,0},
        z_layer = .ui
    )

    mouse_pos := mouse_pos_in_world_space()
    button_bounds := aabb_make(button_pos, bounds_act, Pivot.center_center)
    hover := aabb_contains(button_bounds, mouse_pos)

    if hover && can_afford && key_just_pressed(.LEFT_MOUSE) {
        unlock_skill(&gs.skills_system, skill)
    }

    draw_text(
        button_pos,
        "Unlock",
        col = Colors.text * v4{1,1,1,1},
        pivot = .center_center,
        z_layer = .ui
    )
}

render_skill_entry :: proc(skill: ^Skill, pos: Vector2, system: ^Skills_System) {
    is_active := system.active_skill == skill
    bg_color := is_active ? v4{0.3, 0.3, 0.3, 0.8} : v4{0.2, 0.2, 0.2, 0.8}

    entry_size := v2{280, 70}
    draw_rect_aabb(pos, entry_size, col = bg_color, z_layer = .ui)

    if skill.is_unlocked {
        name_pos := pos + v2{5, 5}
        draw_text(name_pos, fmt.tprintf("%s (Level %d)", skill.name, skill.level), z_layer = .ui)

        bar_pos := name_pos + v2{0, -20}
        bar_width := 180.0
        bar_height := 10.0
        draw_rect_aabb(bar_pos, v2{auto_cast bar_width, auto_cast bar_height}, col = v4{0.1, 0.1, 0.1, 1}, z_layer = .ui)
        xp_ratio := f32(skill.current_xp) / f32(skill.xp_to_next_level)
        draw_rect_aabb(bar_pos, v2{auto_cast bar_width * xp_ratio, auto_cast bar_height}, col = v4{0, 0.8, 0.2, 1}, z_layer = .ui)

        if !is_active {
            button_pos := pos + v2{200, 5}
            button_size := v2{70, 25}
            draw_rect_aabb(button_pos, button_size, col = v4{0.3, 0.5, 0.3, 1}, z_layer = .ui)
            text_pos := button_pos + v2{10, 5}
            draw_text(text_pos, "Select", z_layer = .ui)

            mouse_pos := mouse_pos_in_world_space()
            button_bounds := aabb_make(button_pos, button_size, Pivot.bottom_left)
            if aabb_contains(button_bounds, mouse_pos) && key_just_pressed(.LEFT_MOUSE) {
                system.active_skill = skill
            }
        }

        bonus_text_pos := pos + v2{5, -45}
        bonus := calculate_skill_bonus(skill) * 100
        draw_text(bonus_text_pos, fmt.tprintf("Bonus: +%.1f%%", bonus), z_layer = .ui)
    } else {
        name_pos := pos + v2{5, 5}
        draw_text(name_pos, skill.name, z_layer = .ui)

        cost := get_skill_cost(skill.type)
        cost_pos := pos + v2{5, -20}
        draw_text(cost_pos, fmt.tprintf("Cost: %d gold", cost), z_layer = .ui)

        can_afford := system.gold >= cost
        button_color := can_afford ? v4{0.3, 0.5, 0.3, 1} : v4{0.5, 0.3, 0.3, 1}

        button_pos := pos + v2{200, 5}
        button_size := v2{70, 25}
        draw_rect_aabb(button_pos, button_size, col = button_color, z_layer = .ui)
        text_pos := button_pos + v2{10, 5}
        draw_text(text_pos, "Unlock", z_layer = .ui)

        if can_afford {
            mouse_pos := mouse_pos_in_world_space()
            button_bounds := aabb_make(button_pos, button_size, Pivot.bottom_left)
            if aabb_contains(button_bounds, mouse_pos) && key_just_pressed(.LEFT_MOUSE) {
                unlock_skill(system, skill)
            }
        }
    }
}

render_active_skill_ui :: proc(skill: ^Skill) {
    text_pos := v2{-620, 300}
    draw_text(text_pos, fmt.tprintf("%s (Level %d)", skill.name, skill.level), scale = 1.5, z_layer = .ui)

    bar_width := 200.0
    bar_height := 20.0
    bar_pos := text_pos + v2{0, -30}

    draw_rect_aabb(bar_pos, v2{auto_cast bar_width, auto_cast bar_height}, col = v4{0.2, 0.2, 0.2, 1}, z_layer = .ui)

    xp_ratio := f32(skill.current_xp) / f32(skill.xp_to_next_level)
    draw_rect_aabb(bar_pos, v2{auto_cast bar_width * xp_ratio, auto_cast bar_height}, col = v4{0, 0.8, 0.2, 1}, z_layer = .ui)

    xp_text_pos := bar_pos + v2{0, -20}
    draw_text(xp_text_pos, fmt.tprintf("XP: %d / %d", skill.current_xp, skill.xp_to_next_level), z_layer = .ui)

    bonus_text_pos := xp_text_pos + v2{0, -20}
    bonus := calculate_skill_bonus(skill) * 100
    draw_text(bonus_text_pos, fmt.tprintf("Bonus: +%.1f%%", bonus), z_layer = .ui)
}

render_skill_menu_button :: proc() {
    cfg := gs.ui_config.skills.button
    button_pos := v2{0, cfg.pos_y}
    button_size := v2{cfg.size_x, cfg.size_y} * gs.ui.skills_button_scale

    draw_sprite_with_size(
        button_pos,
        button_size,
        cfg.sprite,
        pivot = .center_center,
        xform = xform_scale(v2{gs.ui.skills_button_scale, gs.ui.skills_button_scale}),
        z_layer = .ui,
    )
}

calculate_gold_gain :: proc(base_gold: int) -> int {
    final_gold := f32(base_gold)

    for &skill in gs.skills_system.advanced_skills {
        if skill.is_unlocked && skill.type == .war_preparation {
            bonus := calculate_skill_bonus(&skill)
            final_gold *= (1.0 + bonus)
            break
        }
    }

    return int(final_gold)
}

//
// :hotreload
init_ui_hot_reload :: proc() -> UI_Hot_Reload {
   hr := UI_Hot_Reload{
       config_path = "./res_workbench/ui_config.json",
       config = UI_Config{
           skills = Skills_UI_Config{
               menu = {
                   pos_y = game_res_h * 0.25,
                   size_x = 400,
                   size_y = 500,
                   background_sprite = .skills_panel_bg,
               },
               button = {
                   pos_y = game_res_h * 0.45,
                   size_x = 48,
                   size_y = 48,
                   sprite = .skills_button,
               },
               next_skill = {
                   offset_x = 0,
                   offset_y = 150,
                   panel_size_x = 320,
                   panel_size_y = 100,
                   title_offset_x = -140,
                   title_offset_y = 20,
                   name_offset_x = -140,
                   name_offset_y = -5,
                   cost_offset_x = -140,
                   cost_offset_y = -30,
                   button_offset_x = 120,
                   button_offset_y = 0,
                   button_size_x = 100,
                   button_size_y = 30,
               },
               tooltip = {
                   offset_x = 0,
                   offset_y = 30,
                   size_x = 200,
                   size_y = 120,
                   padding_x = 10,
                   padding_y = 10,
                   line_spacing = 20,
               },
           },
       },
   }

   if !os.exists(hr.config_path) {
       save_ui_config(&hr)
   }

   load_ui_config(&hr)
   return hr
}

save_ui_config :: proc(hr: ^UI_Hot_Reload) {
   data, err := json.marshal(hr.config)
   if err != nil {
       log_error("Error marshaling config:", err)
       return
   }
   os.write_entire_file(hr.config_path, data)
}

load_ui_config :: proc(hr: ^UI_Hot_Reload) {
   data, ok := os.read_entire_file(hr.config_path)
   if !ok {
       log_error("Could not read config file")
       return
   }

   err := json.unmarshal(data, &hr.config)
   if err != nil {
       log_error("Error unmarshaling config:", err)
       return
   }

   if file_info, err := os.stat(hr.config_path); err == 0 {
       hr.last_modified_time = file_info.modification_time
   }
}

check_and_reload :: proc(hr: ^UI_Hot_Reload) {
   if file_info, err := os.stat(hr.config_path); err == 0 {
       if time.duration_seconds(time.diff(hr.last_modified_time, file_info.modification_time)) > 0 {
           load_ui_config(hr)
           log_error("Reloaded UI configuration")
       }
   }
}

//
// :floating text

Floating_Text :: struct {
    pos: Vector2,
    text: string,
    lifetime: f32,
    alpha: f32,
    scale: f32,
    color: Vector4,
    velocity: Vector2,
    target_scale: f32,
    current_scale: f32,
}

MAX_FLOATING_TEXTS :: 32
floating_texts: [MAX_FLOATING_TEXTS]Floating_Text

add_floating_text :: proc(pos: Vector2, text: string, color := v4{0,1,0,1}) {
    for &ft in floating_texts {
        if ft.lifetime <= 0 {
            text_copy := strings.clone(text)

            ft = Floating_Text{
                pos = pos,
                text = text_copy,
                lifetime = 1.2,
                alpha = 1.0,
                scale = 1.0,
                target_scale = 1.5,
                current_scale = 1.0,
                color = color,
                velocity = v2{0, 25},
            }
            return
        }
    }
}

add_floating_text_params :: proc(
    pos: Vector2,
    text: string,
    color := v4{0,1,0,1},
    scale:f32 = 1.0,
    target_scale:f32 = 1.0,
    lifetime:f32 = 1.0,
    alpha:f32 = 1.0,
    velocity:Vector2 = v2{0, 25},
)
    {
    for &ft in floating_texts {
        if ft.lifetime <= 0 {
            text_copy := strings.clone(text)

            ft = Floating_Text{
                pos = pos,
                text = text_copy,
                lifetime = lifetime,
                alpha = alpha,
                scale = scale,
                target_scale = target_scale,
                current_scale = scale,
                color = color,
                velocity = velocity,
            }
            return
        }
    }
}

update_floating_texts :: proc(dt: f32) {
    for &ft in floating_texts {
        if ft.lifetime > 0 {
            ft.lifetime -= dt
            ft.pos += ft.velocity * dt

            animate_to_target_f32(&ft.current_scale, ft.target_scale, dt * 8)
            ft.scale = ft.current_scale

            if ft.lifetime < 0.5 {
                ft.alpha = ease.cubic_in(ft.lifetime / 0.5)
            }

            if ft.lifetime <= 0 {
                delete(ft.text)
                ft = Floating_Text{}
            }
        }
    }
}

render_floating_texts :: proc() {
    for ft in floating_texts {
        if ft.lifetime > 0 {
            draw_text(
                ft.pos,
                ft.text,
                col = ft.color * v4{1,1,1,ft.alpha},
                scale = auto_cast ft.scale,
                z_layer = .ui,
            )
        }
    }
}

// :text handler
Text_Bounds :: struct {
    width: f32,
    height: f32,
}

get_text_dimensions :: proc(text: string, scale: f32) -> Vector2 {
    using stbtt

    total_size : v2
    for char, i in text {
        advance_x: f32
        advance_y: f32
        q: aligned_quad
        GetBakedQuad(&font.char_data[0], font_bitmap_w, font_bitmap_h, cast(i32)char - 32, &advance_x, &advance_y, &q, false)

        size := v2{ abs(q.x0 - q.x1), abs(q.y0 - q.y1) }

        if i == len(text)-1 {
            total_size.x += size.x
        } else {
            total_size.x += advance_x
        }

        total_size.y = max(total_size.y, -q.y0)
    }

    return total_size * scale
}

wrap_text :: proc(text: string, bounds: Text_Bounds, scale: f32) -> []string {
    if len(text) == 0 do return nil

    lines := make([dynamic]string)
    words := strings.split(text, " ")
    defer delete(words)

    current_line := strings.builder_make()
    defer strings.builder_destroy(&current_line)

    line_start := true

    for word in words {
        test_line: string
        if line_start {
            test_line = word
        } else {
            test_line = fmt.tprintf("%s %s", strings.to_string(current_line), word)
        }

        dims := get_text_dimensions(test_line, scale)

        if dims.x <= bounds.width {
            if line_start {
                strings.write_string(&current_line, word)
            } else {
                strings.write_string(&current_line, " ")
                strings.write_string(&current_line, word)
            }
            line_start = false
        } else {
            if !line_start {
                append(&lines, strings.clone(strings.to_string(current_line)))
                strings.builder_reset(&current_line)
                strings.write_string(&current_line, word)
            } else {
                chars := strings.split(word, "")
                defer delete(chars)

                current_part := strings.builder_make()
                defer strings.builder_destroy(&current_part)

                for char in chars {
                    test_str := fmt.tprintf("%s%s", strings.to_string(current_part), char)
                    test_dims := get_text_dimensions(test_str, scale)

                    if test_dims.x > bounds.width && strings.builder_len(current_part) > 0 {
                        append(&lines, strings.clone(strings.to_string(current_part)))
                        strings.builder_reset(&current_part)
                    }

                    strings.write_string(&current_part, char)
                }

                if strings.builder_len(current_part) > 0 {
                    strings.write_string(&current_line, strings.to_string(current_part))
                }
            }
            line_start = false
        }
    }

    if strings.builder_len(current_line) > 0 {
        append(&lines, strings.clone(strings.to_string(current_line)))
    }

    return lines[:]
}

draw_wrapped_text :: proc(pos: Vector2, text: string, bounds: Text_Bounds, col := COLOR_WHITE, scale := f32(1.0), pivot := Pivot.bottom_left, z_layer := ZLayer.nil) {
    lines := wrap_text(text, bounds, scale)
    defer delete(lines)

    line_height := get_text_dimensions("M", scale).y * 1.2

    total_height := line_height * f32(len(lines))
    offset := v2{0, 0}

    #partial switch pivot {
        case .center_center, .center_left, .center_right:
            offset.y = total_height * 0.5
        case .top_left, .top_center, .top_right:
            offset.y = total_height
    }

    for i := 0; i < len(lines); i += 1 {
        line_pos := pos - v2{0, f32(i) * line_height} - offset
        draw_text(line_pos, lines[i], col = col, scale = auto_cast scale, pivot = pivot, z_layer = z_layer)
    }
}