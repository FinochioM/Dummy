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
import rand "core:math/rand"

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
	})
}

init :: proc "c" () {
	using linalg, fmt
	context = runtime.default_context()

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

    gs.skills_system = init_skills_system()
    gs.quests_system = init_quests_system()

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

// might do something with these later on
loggie :: fmt.println // log is already used........
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

//
// :RENDER STUFF
//
// API ordered highest -> lowest level

draw_sprite :: proc(pos: Vector2, img_id: Image_Id, pivot:= Pivot.bottom_left, xform := Matrix4(1), color_override:= v4{0,0,0,0}, z_layer := ZLayer.nil) {
	image := images[img_id]
	size := v2{auto_cast image.width, auto_cast image.height}

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
    }
}
draw_frame : Draw_Frame

ZLayer :: enum u8{
    nil,
    background,
    player,
    foreground,
    ui,
}

Coord_Space :: struct {
    proj: Matrix4,
    camera: Matrix4,
}

set_draw_frame :: proc(coord: Coord_Space) {
    draw_frame.coord_space = coord
}

@(deferred_out=set_draw_frame)
push_coord_space :: proc(coord: Coord_Space) -> Coord_Space {
    og := draw_frame.coord_space
    draw_frame.coord_space = coord
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
		tex_index = 255 // bypasses texture sampling
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

	verts[0].z_layer = u8(z_layer)
	verts[1].z_layer = u8(z_layer)
	verts[2].z_layer = u8(z_layer)
	verts[3].z_layer = u8(z_layer)
}

//
// :IMAGE STUFF
//
Image_Id :: enum {
	nil,
	player_move1,
	background,
	foreground,
	dummy,
	arrow,
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
draw_text :: proc(pos: Vector2, text: string, scale:= 1.0, z_layer := ZLayer.nil) {
	using stbtt

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

		uv := v4{ q.s0, q.t1, q.s1, q.t0 }

		xform := Matrix4(1)
		xform *= xform_translate(pos)
		xform *= xform_scale(v2{auto_cast scale, auto_cast scale})
		xform *= xform_translate(offset_to_render_at)
		draw_rect_xform(xform, size, uv=uv, img_id=font.img_id, z_layer = z_layer)

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
	skills_system: Skills_System,
    quests_system: Quests_System,
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

	if key_just_pressed(.F11) {
		sapp.toggle_fullscreen()
	}

	if gs.ticks == 0 {
	    // CREATE PLAYER
		en := entity_create()
		setup_player(en)
		gs.player_handle = entity_to_handle(en^)
	}

	draw_frame.coord_space.proj = matrix_ortho3d_f32(window_w * -0.5, window_w * 0.5, window_h * -0.5, window_h * 0.5, -1, 1)

	draw_frame.coord_space.camera = Matrix4(1)
	draw_frame.coord_space.camera *= xform_scale(f32(window_h) / f32(game_res_h))

	for &en in gs.entities {
		en.frame = {}
	}

	player := get_player()

    // UPDATE ENTITIES
	for &en in gs.entities {
        if .allocated in en.flags {
            #partial switch en.kind {
                case .player: update_player(&en, f32(dt))
                case .arrow: update_arrow(&en, f32(dt))
            }
        }
	}

	check_spawn_button()
	update_quests_system(&gs.quests_system, f32(dt))

	gs.ticks += 1
}

render :: proc() {
	using linalg

    draw_rect_aabb(v2{ game_res_w * -0.5, game_res_h * -0.5}, v2{game_res_w, game_res_h}, img_id=.background, z_layer = .background)
    draw_rect_aabb(v2{ game_res_w * -0.5, game_res_h * -0.5}, v2{game_res_w, game_res_h}, img_id=.foreground, z_layer = .foreground)

    if !has_active_dummy() {
        button_pos := v2{-BUTTON_WIDTH / 2, 200}

        draw_rect_aabb(
            button_pos,
            v2{BUTTON_WIDTH, BUTTON_HEIGHT},
            col = v4{0.2, 0.2, 0.2, 1},
            z_layer = .ui,
        )

        text_pos := button_pos + v2{BUTTON_WIDTH / 2 - 70, BUTTON_HEIGHT / 2 - 10}
        draw_text(text_pos, "Spawn Dummy", scale = 2.0, z_layer = .ui)
    }

	for &en in gs.entities {
		if .allocated in en.flags {
			#partial switch en.kind {
				case .player: {
				    draw_player_at_pos(en, v2{-440, -320})
				}
				case .dummy: {
				    draw_dummy_at_pos(en)
				}
				case .arrow: {
				    draw_arrow_at_pos(&en)
				}
			}
		}
	}

    if gs.skills_system.is_unlocked {
        render_skill_menu_button()
        render_quest_menu_button()

        if gs.skills_system.menu_open {
            render_skills_ui()
        } else if gs.quests_system.menu_open {
            render_quests_ui()
        }
    }

    draw_text(v2{-200, 100}, "Dummy", scale = 2.0, z_layer = .ui)

	gs.ticks += 1
}

draw_player_at_pos :: proc(en: Entity, pos: Vector2) {
    xform := Matrix4(1)
    xform *= xform_scale(v2{3.2, 3.2})

    draw_sprite(pos, .player_move1, pivot=.bottom_center, xform = xform, z_layer = .player)
}

draw_dummy_at_pos :: proc(en: Entity){
    xform := Matrix4(1)
    xform *= xform_scale(v2{2.5,2.5})

    health_percent := en.health / en.max_health

    draw_sprite(en.pos, .dummy, pivot = .bottom_center, xform = xform,z_layer = .player)
    if en.health < en.max_health {
        bar_width := 64.0
        bar_height := 8.0
        health_ratio := en.health / en.max_health

        bar_pos := en.pos + v2{auto_cast -bar_width / 2, 100}
        draw_rect_aabb(bar_pos, v2{auto_cast bar_width, auto_cast bar_height}, col=v4{0.243,0.243,0.259,1}, z_layer = .ui)

        draw_rect_aabb(bar_pos, v2{auto_cast bar_width * health_ratio, auto_cast bar_height * 0.5}, col=v4{0,1,0,1}, z_layer = .ui)
    }
}

draw_arrow_at_pos :: proc(en: ^Entity){
    xform := Matrix4(1)
    xform *= xform_translate(en.pos)
    xform *= xform_rotate(en.rotation)
    xform *= xform_scale(v2{2.0, 2.0})

    draw_sprite(v2{0,0}, .arrow, pivot = .center_center, xform = xform, z_layer = .player)
}

mouse_pos_in_world_space :: proc() -> Vector2 {
	if draw_frame.coord_space.proj == {} {
		log_error("no projection matrix set yet")
	}

	mouse := v2{app_state.input_state.mouse_x, app_state.input_state.mouse_y}
	ndc_x := (mouse.x / (f32(window_w) * 0.5)) - 1.0;
	ndc_y := (mouse.y / (f32(window_h) * 0.5)) - 1.0;
	ndc_y *= -1

	mouse_ndc := v2{ndc_x, ndc_y}

	mouse_world :v4= v4{mouse_ndc.x, mouse_ndc.y, 0, 1}

	mouse_world *= linalg.inverse(draw_frame.coord_space.proj)
	mouse_world *= linalg.inverse(draw_frame.coord_space.camera)

	return mouse_world.xy
}

//
// :dummies
DUMMY_MAX_HEALTH :: 100.0
ARROW_DAMAGE :: 20.0

spawn_dummy :: proc(position: Vector2) -> ^Entity {
    dummy := entity_create()
    if dummy == nil do return nil

    setup_dummy(dummy)
    dummy.pos = position

    return dummy
}

//
// :player
update_player :: proc(player: ^Entity, dt: f32) {
    player.shoot_cooldown -= dt

    if player.shoot_cooldown <= 0 {
        target := find_random_target()
        if target != nil {
            shoot_arrow(player, target)

            base_cooldown := SHOOT_COOLDOWN
            if system := &gs.skills_system; system.is_unlocked {
                for &skill in system.skills {
                    if skill.is_unlocked && skill.type == .speed_boost {
                        speed_bonus := calculate_skill_bonus(&skill)
                        base_cooldown *= (1.0 - f64(speed_bonus))
                    }
                }
            }
            player.shoot_cooldown = f32(base_cooldown)
        }
    }
}

find_random_target :: proc() -> ^Entity{
    targets: [dynamic]^Entity
    defer delete(targets)

    for &en in gs.entities{
        if .allocated in en.flags && en.kind == .dummy{
            append(&targets, &en)
        }
    }

    if len(targets) > 0 {
        return targets[rand.int_max(len(targets))]
    }

    return nil
}

//
// :arrows
Arrow_Data :: struct {
    velocity: Vector2,
    target_pos: Vector2,
    lifetime: f32,
}

SHOOT_COOLDOWN :: 1.5
ARROW_SPEED :: 1200.0
ARROW_LIFETIME :: 2.0
ACCURACY_VARIANCE :: 20.0
GRAVITY_EFFECT :: 2000.0

update_arrow :: proc(e: ^Entity, dt: f32) {
    e.arrow_data.velocity.y -= GRAVITY_EFFECT * dt

    old_pos := e.pos
    e.pos += e.arrow_data.velocity * dt

    e.arrow_data.lifetime -= dt
    if e.arrow_data.lifetime <= 0 {
        entity_destroy(e)
        return
    }

    arrow_size := v2{10, 1}
    arrow_aabb := aabb_make(e.pos, arrow_size, .center_center)

    for &target in gs.entities {
        if .allocated not_in target.flags || target.kind != .dummy {
            continue
        }

        dummy_size := v2{32, 32}
        target.aabb = aabb_make(target.pos, dummy_size, .bottom_center)

        if aabb_collide(arrow_aabb, target.aabb){
            damage_entity(&target, ARROW_DAMAGE)
            entity_destroy(e)
            return
        }
    }

    angle := math.atan2(e.arrow_data.velocity.y, e.arrow_data.velocity.x)
    e.rotation = math.to_degrees(angle)
}

shoot_arrow :: proc(player: ^Entity, target: ^Entity){
    arrow := entity_create()
    if arrow == nil do return

    shoot_pos := v2{-440, -320} + v2{30, 40}
    setup_arrow(arrow, shoot_pos, target.pos + v2{0, 15})
}

//
// :entity
//

Entity_Flags :: enum {
	allocated,
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
	health: f32,
	max_health: f32,
	aabb: Vector4,
	img_id: Image_Id,
}

entity_data: [Entity_Kind]Entity

Entity_Handle :: struct {
    id: u64,
    index: int,
}

damage_entity :: proc(e: ^Entity, base_damage: f32) {
    damage := base_damage
    crit_occurred := false

    if system := &gs.skills_system; system.is_unlocked {
        for &skill in system.skills {
            if skill.is_unlocked {
                #partial switch skill.type {
                    case .strength_boost:
                        strength_bonus := calculate_skill_bonus(&skill)
                        damage *= (1.0 + strength_bonus)
                    case .critical_boost:
                        crit_chance := calculate_skill_bonus(&skill)
                        if rand.float32() < crit_chance {
                            damage *= 2.0
                            crit_occurred = true
                        }
                }
            }
        }
    }

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

entity_destroy :: proc(entity: ^Entity) {
    if entity.kind == .dummy {
        gs.skills_system.dummies_killed += 1
        check_skills_unlock(&gs.skills_system)

        if gs.skills_system.is_unlocked {
            add_xp_to_active_skill(&gs.skills_system, 50)
        }
    }

	mem.set(entity, 0, size_of(Entity))
}


//
// :setups

setup_player :: proc(e: ^Entity) {
	e.kind = .player
    e.flags |= { .allocated }
}

setup_dummy :: proc(e: ^Entity){
    e.kind = .dummy
    e.flags |= { .allocated }
    e.health = DUMMY_MAX_HEALTH
    e.max_health = DUMMY_MAX_HEALTH

    dummy_size := v2{32, 32}
    e.aabb = aabb_make(e.pos, dummy_size, .bottom_center)
}

setup_arrow :: proc(e: ^Entity, start_pos: Vector2, target_pos: Vector2){
    e.kind = .arrow
    e.flags |= {.allocated}
    e.pos = start_pos

    direction := target_pos - start_pos
    distance := linalg.length(direction)
    flight_time := distance / ARROW_SPEED

    target_with_variance := target_pos + v2{
        rand.float32_range(-ACCURACY_VARIANCE, ACCURACY_VARIANCE),
        rand.float32_range(-ACCURACY_VARIANCE, ACCURACY_VARIANCE),
    }

    direction_to_target := target_with_variance - start_pos
    direction_normalized := direction_to_target / linalg.length(direction_to_target)

    base_velocity := direction_normalized * ARROW_SPEED
    vertical_boost := GRAVITY_EFFECT * flight_time * 0.5

    e.arrow_data = Arrow_Data {
        velocity = base_velocity + v2{0, vertical_boost},
        target_pos = target_pos,
        lifetime = ARROW_LIFETIME,
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
        anim.current_frame += 1

        if anim.current_frame >= len(anim.frames) {
            if anim.loops {
                anim.current_frame = 0
            } else {
                anim.current_frame = len(anim.frames) - 1
                anim.state = .Stopped
                return true
            }
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

draw_animated_sprite :: proc(pos: Vector2, anim: ^Animation, pivot := Pivot.bottom_left, xform := Matrix4(1), color_override := v4{0,0,0,0}){
    if anim == nil do return
    current_frame := get_current_frame(anim)
    draw_sprite(pos, current_frame, pivot, xform, color_override)
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

reset_and_play_animation :: proc(collection: ^Animation_Collection, name: string, speed: f32 = 1.0){
    if collection == nil do return

    if anim, ok := &collection.animations[name]; ok{
        anim.current_frame = 0
        anim.frame_timer = 0
        anim.state = .Playing
        anim.loops = false
        adjust_animation_to_speed(anim, speed)

        collection.current_animation = name
    }
}

update_current_animation :: proc(collection: ^Animation_Collection, delta_t: f32) {
    if collection.current_animation != "" {
        if anim, ok := &collection.animations[collection.current_animation]; ok {
            animation_finished := update_animation(anim, delta_t)
            if animation_finished && collection.current_animation == "attack"{
                play_animation_by_name(collection, "idle")
            }
        }
    }
}

draw_current_animation :: proc(collection: ^Animation_Collection, pos: Vector2, pivot := Pivot.bottom_left, xform := Matrix4(1), color_override := v4{0,0,0,0}) {
    if collection == nil || collection.current_animation == "" {
        return
    }
    if anim, ok := &collection.animations[collection.current_animation]; ok {
        draw_animated_sprite(pos, anim, pivot, xform, color_override)
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

aabb_make :: proc(pos: Vector2, size: Vector2, pivot: Pivot) -> Vector4{
    half_size := size * 0.5

    offset := -scale_from_pivot(pivot) * size

    min := pos + offset
    max := min + size

    return Vector4{min.x, min.y, max.x, max.y}
}

aabb_collide :: proc(a, b: Vector4) ->bool {
    return !(a.z < b.x || a.x > b.z || a.w < b.y || a.y > b.w)
}

aabb_contains :: proc(aabb: Vector4, p: Vector2) -> bool {
    return (p.x >= aabb.x) && (p.x <= aabb.z) &&
           (p.y >= aabb.y) && (p.y <= aabb.w)
}

//
// :ui & control

BUTTON_WIDTH :: 200.0
BUTTON_HEIGHT :: 50.0

has_active_dummy :: proc() -> bool {
    for &en in gs.entities {
        if .allocated in en.flags && en.kind == .dummy {
            return true
        }
    }
    return false
}

check_spawn_button :: proc() {
    if has_active_dummy() {
        return
    }

    if draw_frame.coord_space.proj == {} {
        return
    }

    if key_just_pressed(.LEFT_MOUSE) {
        fmt.println("Left mouse just pressed")
        fmt.println("Raw mouse pos:", app_state.input_state.mouse_x, app_state.input_state.mouse_y)
        world_pos := mouse_pos_in_world_space()
        fmt.println("World pos:", world_pos)
    }

    button_pos := v2{-BUTTON_WIDTH/2, 200}
    mouse_pos := mouse_pos_in_world_space()

    button_bounds := v4{
        button_pos.x,
        button_pos.y,
        button_pos.x + BUTTON_WIDTH,
        button_pos.y + BUTTON_HEIGHT,
    }

    if aabb_contains(button_bounds, mouse_pos) && key_just_pressed(.LEFT_MOUSE) {
        spawn_dummy(v2{300, -320})
    }
}

//
// :quests
QUEST_TICK_TIME :: 1.5

Quest_Type :: enum {
    nil,
    gold_generation,
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
}

Quests_System :: struct {
    quests: [dynamic]Quest,
    active_quest: ^Quest,
    timer: f32,
    menu_open: bool,
}

init_quests_system :: proc() -> Quests_System {
    system := Quests_System {
        quests = make([dynamic]Quest),
        active_quest = nil,
        timer = 0,
        menu_open = false,
    }

    gold_quest := Quest{
        type = .gold_generation,
        name = "Gold Generation",
        description ="Generates gold over time",
        level = 1,
        is_unlocked = false,
        cooldown = QUEST_TICK_TIME,
        required_skill = .xp_boost,
        required_skill_levels = {5, 8, 12, 15, 20},
        gold_per_tick = {1, 2, 3, 4, 5},
    }

    append(&system.quests, gold_quest)

    return system
}

update_quests_system :: proc(system: ^Quests_System, dt: f32) {
    if system.active_quest == nil {
        return
    }

    system.timer -= dt
    if system.timer <= 0 {
        system.timer = QUEST_TICK_TIME
        give_quest_rewards(system.active_quest)
    }
}

give_quest_rewards :: proc(quest: ^Quest) {
    if quest == nil {
        return
    }

    #partial switch quest.type {
        case .gold_generation:
            gs.skills_system.gold += quest.gold_per_tick[quest.level - 1]
    }
}

check_quest_unlocks :: proc(skill: ^Skill) {
    if skill == nil {
        return
    }

    for &quest in gs.quests_system.quests {
        if !quest.is_unlocked && quest.required_skill == skill.type {
            if skill.level >= quest.required_skill_levels[0] {
                quest.is_unlocked = true
            }
        }
    }
}

render_quests_ui :: proc() {
    system := &gs.quests_system
    if system == nil {
        return
    }

    menu_pos := v2{-620, 260}
    menu_size := v2{300, 400}
    draw_rect_aabb(menu_pos, menu_size, col = v4{0.1, 0.1, 0.1, 0.9}, z_layer = .ui)

    quest_y := menu_pos.y - 50
    for &quest in system.quests {
        if quest.is_unlocked {
            render_quest_entry(&quest, v2{menu_pos.x + 10, quest_y}, system)
            quest_y -= 80
        }
    }
}

render_quest_entry :: proc(quest: ^Quest, pos: Vector2, system: ^Quests_System) {
    is_active := system.active_quest == quest
    bg_color := is_active ? v4{0.3, 0.3, 0.3, 0.8} : v4{0.2, 0.2, 0.2, 0.8}

    entry_size := v2{280, 70}
    draw_rect_aabb(pos, entry_size, col = bg_color, z_layer = .ui)

    name_pos := pos + v2{5, 5}
    draw_text(name_pos, fmt.tprintf("%s (Level %d)", quest.name, quest.level), z_layer = .ui)

    reward_pos := pos + v2{5, -20}
    draw_text(reward_pos, fmt.tprintf("Gold per tick: %d", quest.gold_per_tick[quest.level - 1]), z_layer = .ui)

    if !is_active {
        button_pos := pos + v2{200, 5}
        button_size := v2{70, 25}
        draw_rect_aabb(button_pos, button_size, col = v4{0.3, 0.5, 0.3, 1}, z_layer = .ui)
        text_pos := button_pos + v2{10, 5}
        draw_text(text_pos, "Select", z_layer = .ui)

        mouse_pos := mouse_pos_in_world_space()
        button_bounds := aabb_make(button_pos, button_size, .bottom_left)
        if aabb_contains(button_bounds, mouse_pos) && key_just_pressed(.LEFT_MOUSE) {
            system.active_quest = quest
            system.timer = QUEST_TICK_TIME
        }
    }

    if quest.level < 5 {
        next_level_pos := pos + v2{5, -45}
        req_skill_level := quest.required_skill_levels[quest.level]
        draw_text(next_level_pos, fmt.tprintf("Next level at %s level %d",
            gs.skills_system.skills[quest.required_skill].name, req_skill_level), z_layer = .ui)
    }
}

render_active_quest_ui :: proc(quest: ^Quest) {
    if quest == nil {
        return
    }

    text_pos := v2{-620, 240}
    draw_text(text_pos, fmt.tprintf("%s (Level %d)", quest.name, quest.level), scale = 1.2, z_layer = .ui)

    reward_pos := text_pos + v2{0, -25}
    draw_text(reward_pos, fmt.tprintf("Gold per tick: %d", quest.gold_per_tick[quest.level - 1]), z_layer = .ui)
}

render_quest_menu_button :: proc() {
    button_pos := v2{-620, 280}
    button_size := v2{120, 30}
    draw_rect_aabb(button_pos, button_size, col = v4{0.2, 0.2, 0.2, 1}, z_layer = .ui)
    text_pos := button_pos + v2{10, 8}
    draw_text(text_pos, "Quests Menu", z_layer = .ui)

    mouse_pos := mouse_pos_in_world_space()
    button_bounds := aabb_make(button_pos, button_size, .bottom_left)
    if aabb_contains(button_bounds, mouse_pos) && key_just_pressed(.LEFT_MOUSE) {
        gs.quests_system.menu_open = !gs.quests_system.menu_open
        if gs.quests_system.menu_open {
            gs.skills_system.menu_open = false
        }
    }
}

//
// :skills
Skill_Type :: enum {
    nil,
    xp_boost,
    strength_boost,
    speed_boost,
    critical_boost,
}

Skill :: struct {
    type: Skill_Type,
    name: string,
    level: int,
    current_xp: int,
    xp_to_next_level: int,
    description: string,
    is_unlocked: bool,
}

Skills_System :: struct {
    skills: [dynamic]Skill,
    active_skill: ^Skill,
    dummies_killed: int,
    is_unlocked: bool,
    gold: int,
    menu_open: bool,
}

SKILL_COSTS :: [Skill_Type]int {
    .nil = 0,
    .xp_boost = 0,
    .strength_boost = 300,
    .speed_boost = 500,
    .critical_boost = 800,
}

init_skills_system :: proc() -> Skills_System {
    system := Skills_System{
        skills = make([dynamic]Skill),
        active_skill = nil,
        dummies_killed = 0,
        is_unlocked = false,
        gold = 10000,
        menu_open = false,
    }

    skill_data := []struct{type: Skill_Type, name, desc: string}{
        {.xp_boost, "Experience Mastery", "Increases experience gained from destroying dummies by 5% per level"},
        {.strength_boost, "Strength Mastery", "Increases arrow damage by 5% per level"},
        {.speed_boost, "Speed Mastery", "Increases attack speed by 5% per level"},
        {.critical_boost, "Critical Mastery", "Increases critical hit chance by 1% per level"},
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

    return system
}

calculate_xp_boost :: proc(skill: ^Skill) -> f32{
    if skill == nil || skill.type != .xp_boost {
        return 1.0
    }

    base_boost := 0.05
    level_boost := 0.05 * f32(skill.level - 1)
    return 1.0 + f32(base_boost) + f32(level_boost)
}

calculate_skill_bonus :: proc(skill: ^Skill) -> f32 {
    if skill == nil {
        return 0.0
    }

    base_bonus: f32
    #partial switch skill.type {
        case .xp_boost: base_bonus = 0.05
        case .strength_boost: base_bonus = 0.05
        case .speed_boost: base_bonus = 0.05
        case .critical_boost: base_bonus = 0.01
        case: return 0.0
    }

    return base_bonus * f32(skill.level)
}

add_xp_to_active_skill :: proc(system: ^Skills_System, base_xp: int) {
    if system.active_skill == nil {
        return
    }

    xp_multiplier := calculate_xp_boost(system.active_skill)
    total_xp := int(f32(base_xp) * xp_multiplier)

    prev_level := system.active_skill.level
    system.active_skill.current_xp += total_xp

    for system.active_skill.current_xp >= system.active_skill.xp_to_next_level {
        system.active_skill.current_xp -= system.active_skill.xp_to_next_level
        system.active_skill.level += 1
        system.active_skill.xp_to_next_level = int(f32(system.active_skill.xp_to_next_level) * 1.5)
    }

    if system.active_skill.level > prev_level {
        check_quest_unlocks(system.active_skill)
    }
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

unlock_skill :: proc(system: ^Skills_System, skill: ^Skill) {
    cost := get_skill_cost(skill.type)
    if system.gold >= cost {
        system.gold -= cost
        skill.is_unlocked = true
        if system.active_skill == nil {
            system.active_skill = skill
        }
    }
}

get_skill_cost :: proc(type: Skill_Type) -> int {
    switch type {
        case .nil: return 0
        case .xp_boost: return 0
        case .strength_boost: return 300
        case .speed_boost: return 500
        case .critical_boost: return 800
    }
    return 0
}

render_skills_ui :: proc() {
    system := &gs.skills_system
    if system == nil || !system.is_unlocked {
        return
    }

    menu_pos := v2{-620, 320}
    menu_size := v2{300, 400}
    draw_rect_aabb(menu_pos, menu_size, col = v4{0.1, 0.1, 0.1, 0.9}, z_layer = .ui)

    gold_pos := menu_pos + v2{10, -30}
    draw_text(gold_pos, fmt.tprintf("Gold: %d", system.gold), scale = 1.2, z_layer = .ui)

    skill_y := menu_pos.y - 70
    for &skill in system.skills {
        render_skill_entry(&skill, v2{menu_pos.x + 10, skill_y}, system)
        skill_y -= 80
    }
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
            button_bounds := aabb_make(button_pos, button_size, .bottom_left)
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
            button_bounds := aabb_make(button_pos, button_size, .bottom_left)
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
    button_pos := v2{-620, 340}
    button_size := v2{120, 30}
    draw_rect_aabb(button_pos, button_size, col = v4{0.2, 0.2, 0.2, 1}, z_layer = .ui)
    text_pos := button_pos + v2{10, 8}
    draw_text(text_pos, "Skills Menu", z_layer = .ui)

    mouse_pos := mouse_pos_in_world_space()
    button_bounds := aabb_make(button_pos, button_size, .bottom_left)
    if aabb_contains(button_bounds, mouse_pos) && key_just_pressed(.LEFT_MOUSE) {
        gs.skills_system.menu_open = !gs.skills_system.menu_open
        if gs.skills_system.menu_open {
            gs.quests_system.menu_open = false
        }
    }
}
