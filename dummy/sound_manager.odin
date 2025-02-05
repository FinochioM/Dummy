package main

import fcore "fmod/core"
import fstudio "fmod/studio"
import fsbank "fmod/fsbank"
import "core:fmt"
import "core:strings"
import "core:thread"
thread_pool: [dynamic]^thread.Thread

Sound_State :: struct {
	system: ^fstudio.SYSTEM,
	core_system: ^fcore.SYSTEM,
	bank: ^fstudio.BANK,
	strings_bank: ^fstudio.BANK,
	master_ch_group : ^fcore.CHANNELGROUP,
	playing: bool,
	paused: bool,
	current_music: ^fstudio.EVENTINSTANCE,
}
sound_st: Sound_State

SOUND_COOLDOWN :: 0.1

init_sound :: proc(){
    using fstudio
    using sound_st

	fmod_error_check(System_Create(&system, fcore.VERSION))

	fmod_error_check(System_Initialize(system, 512, INIT_NORMAL, INIT_NORMAL, nil))

	fmod_error_check(System_LoadBankFile(system, "./res_workbench/fmod_dummy/dummy/Build/Desktop/Master.bank", LOAD_BANK_NORMAL, &bank))
	fmod_error_check(System_LoadBankFile(system, "./res_workbench/fmod_dummy/dummy/Build/Desktop/Master.strings.bank", LOAD_BANK_NORMAL, &strings_bank))
}

play_sound :: proc(name: string) {
    using fstudio
    using sound_st

    event_path := fmt.tprintf("event:/%s", name)

    event_desc: ^EVENTDESCRIPTION
    result := System_GetEvent(system, fmt.ctprint(event_path), &event_desc) // Use event_path instead of name

    if result != .OK {
        fmt.println("Failed to get event:", event_path, "Error:", fcore.error_string(result))
        return
    }

    instance: ^EVENTINSTANCE
    fmod_error_check(EventDescription_CreateInstance(event_desc, &instance))
    fmod_error_check(EventInstance_Start(instance))
}

update_sound :: proc() {
    using fstudio
    using sound_st

    fmod_error_check(System_Update(system))
}

fmod_error_check :: proc(result: fcore.RESULT) {
	assert(result == .OK, fcore.error_string(result))
}

stop_sound :: proc(event: ^fstudio.EVENTINSTANCE) -> bool {
	using sound_st, fstudio
	ok := EventInstance_Stop(event, .STOP_ALLOWFADEOUT)
	return ok == .OK
}

toggle_sound :: proc(sound_st: ^Sound_State) {
    if sound_st.playing {
        if sound_st.current_music != nil {
            stop_sound(sound_st.current_music)
            sound_st.current_music = nil
        }
        sound_st.playing = false
    } else {
        play_background_music(sound_st)
        sound_st.playing = true
    }
}

play_background_music :: proc(sound_st: ^Sound_State) {
    using fstudio
    event_desc: ^EVENTDESCRIPTION
    result := System_GetEvent(sound_st.system, "event:/beat", &event_desc)

    if result == .OK {
        instance: ^EVENTINSTANCE
        fmod_error_check(EventDescription_CreateInstance(event_desc, &instance))
        fmod_error_check(EventInstance_Start(instance))
        sound_st.current_music = instance
    }
}