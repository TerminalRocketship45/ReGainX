"""
Inject model.save() into a running train_recurrent.py process without killing it.

Approach:
  1. Find python312.dll base address in the target process.
  2. Build x64 shellcode that calls PyGILState_Ensure -> PyRun_SimpleString -> PyGILState_Release.
  3. Allocate RWX memory in target, write shellcode + param struct + Python code.
  4. CreateRemoteThread to execute the shellcode.
  5. Poll for a marker file written by the injected code.
"""

import ctypes
import ctypes.wintypes
import struct
import sys
import os
import time

TARGET_PID   = 36612
POLICY_DIR   = r"C:\Users\rohan\Downloads\reGainX_git\policies"
SAVE_PATH    = os.path.join(POLICY_DIR, "policy_brady_deg_recurrent_checkpoint")
DONE_MARKER  = os.path.join(POLICY_DIR, ".checkpoint_done")
ERROR_MARKER = os.path.join(POLICY_DIR, ".checkpoint_error")
PYTHON_DLL   = "python312.dll"

# Python source to execute inside the target process
INJECT_SRC = f"""
import sys, os, traceback

try:
    frames = sys._current_frames()
    model = None
    for tid, frame in frames.items():
        f = frame
        depth = 0
        while f is not None and depth < 200:
            if 'model' in f.f_locals:
                model = f.f_locals['model']
                break
            f = f.f_back
            depth += 1
        if model is not None:
            break

    if model is None:
        raise RuntimeError("model not found in any frame")

    os.makedirs({repr(POLICY_DIR)}, exist_ok=True)
    model.save({repr(SAVE_PATH)})

    with open({repr(DONE_MARKER)}, 'w') as _f:
        _f.write(str(model.num_timesteps))
except Exception:
    with open({repr(ERROR_MARKER)}, 'w') as _f:
        _f.write(traceback.format_exc())
"""
INJECT_CODE = INJECT_SRC.encode("utf-8") + b"\x00"

# x64 shellcode — thread function receives RCX = pointer to param struct:
#   [+0 ]  code_str_ptr          (8 bytes, pointer to null-terminated Python source)
#   [+8 ]  addr PyGILState_Ensure  (8 bytes)
#   [+16]  addr PyRun_SimpleString (8 bytes)
#   [+24]  addr PyGILState_Release (8 bytes)
#   [+32]  gilstate storage        (8 bytes, written by shellcode)
#
# Stack on entry: RSP % 16 == 8  (return address on stack from CreateRemoteThread call)
# After push rbx + push rdi: RSP % 16 == 8 (pushed 16 bytes, net 0 mod 16 => still 8)
# sub rsp, 40: 40 % 16 == 8, so RSP % 16 becomes 0  =>  aligned for calls  ✓
_sc = bytes([
    0x53,                        # push rbx
    0x57,                        # push rdi
    0x48, 0x83, 0xec, 0x28,      # sub  rsp, 40       (shadow + align)
    0x48, 0x8b, 0xf9,            # mov  rdi, rcx      (save param ptr)
    0xff, 0x57, 0x08,            # call [rdi+8]       PyGILState_Ensure()
    0x89, 0x47, 0x20,            # mov  [rdi+32], eax (save gilstate)
    0x48, 0x8b, 0x0f,            # mov  rcx, [rdi]    (code_str_ptr -> arg1)
    0xff, 0x57, 0x10,            # call [rdi+16]      PyRun_SimpleString(rcx)
    0x8b, 0x4f, 0x20,            # mov  ecx, [rdi+32] (gilstate -> arg1)
    0xff, 0x57, 0x18,            # call [rdi+24]      PyGILState_Release(ecx)
    0x48, 0x83, 0xc4, 0x28,      # add  rsp, 40
    0x5f,                        # pop  rdi
    0x5b,                        # pop  rbx
    0xc3,                        # ret
])
SHELLCODE = _sc + b"\x90" * (64 - len(_sc))   # pad to 64 bytes

PROCESS_ALL_ACCESS    = 0x1F0FFF
MEM_COMMIT_RESERVE    = 0x3000
PAGE_EXECUTE_READWRITE = 0x40
MEM_RELEASE           = 0x8000

k32 = ctypes.windll.kernel32
k32.VirtualAllocEx.restype              = ctypes.c_void_p
k32.WriteProcessMemory.argtypes         = [
    ctypes.wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
]
k32.CreateRemoteThread.restype          = ctypes.wintypes.HANDLE
k32.GetModuleHandleW.restype            = ctypes.c_uint64


def remote_module_base(pid: int, name_substr: str) -> int | None:
    TH32CS_SNAPMODULE   = 0x00000008
    TH32CS_SNAPMODULE32 = 0x00000010

    class MODULEENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize",        ctypes.wintypes.DWORD),
            ("th32ModuleID",  ctypes.wintypes.DWORD),
            ("th32ProcessID", ctypes.wintypes.DWORD),
            ("GlblcntUsage",  ctypes.wintypes.DWORD),
            ("ProccntUsage",  ctypes.wintypes.DWORD),
            ("modBaseAddr",   ctypes.c_uint64),
            ("modBaseSize",   ctypes.wintypes.DWORD),
            ("hModule",       ctypes.wintypes.HMODULE),
            ("szModule",      ctypes.c_wchar * 256),
            ("szExePath",     ctypes.c_wchar * 260),
        ]

    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    if snap == ctypes.c_void_p(-1).value:
        return None
    me = MODULEENTRY32W()
    me.dwSize = ctypes.sizeof(MODULEENTRY32W)
    base = None
    try:
        if k32.Module32FirstW(snap, ctypes.byref(me)):
            while True:
                if name_substr.lower() in me.szModule.lower():
                    base = me.modBaseAddr
                    break
                if not k32.Module32NextW(snap, ctypes.byref(me)):
                    break
    finally:
        k32.CloseHandle(snap)
    return base


def main() -> bool:
    for marker in (DONE_MARKER, ERROR_MARKER):
        try:
            os.remove(marker)
        except FileNotFoundError:
            pass

    # Load python312.dll into our process and compute function offsets
    local_dll = ctypes.CDLL(
        r"C:\Users\rohan\anaconda3\envs\exo_s\python312.dll"
    )
    local_base: int = k32.GetModuleHandleW(PYTHON_DLL)
    if not local_base:
        print("ERROR: python312.dll not loaded in this process. "
              "Run this script with exo_s\\python.exe")
        return False

    ensure_addr   = ctypes.cast(local_dll.PyGILState_Ensure,   ctypes.c_void_p).value
    run_addr      = ctypes.cast(local_dll.PyRun_SimpleString,  ctypes.c_void_p).value
    release_addr  = ctypes.cast(local_dll.PyGILState_Release,  ctypes.c_void_p).value

    offset_ensure  = ensure_addr  - local_base
    offset_run     = run_addr     - local_base
    offset_release = release_addr - local_base

    print(f"Local  python312.dll base  : {hex(local_base)}")
    print(f"  PyGILState_Ensure  offset: {hex(offset_ensure)}")
    print(f"  PyRun_SimpleString offset: {hex(offset_run)}")
    print(f"  PyGILState_Release offset: {hex(offset_release)}")

    remote_base = remote_module_base(TARGET_PID, "python312")
    if remote_base is None:
        print(f"ERROR: python312.dll not found in PID {TARGET_PID}")
        return False
    print(f"Remote python312.dll base  : {hex(remote_base)}")

    remote_ensure  = remote_base + offset_ensure
    remote_run     = remote_base + offset_run
    remote_release = remote_base + offset_release

    h_proc = k32.OpenProcess(PROCESS_ALL_ACCESS, False, TARGET_PID)
    if not h_proc:
        print(f"ERROR: OpenProcess({TARGET_PID}) failed — {k32.GetLastError()}")
        return False

    # Memory layout in target:
    #   [0  : 64 ]  shellcode
    #   [64 : 104]  param struct  (5 × uint64 = 40 bytes)
    #   [104: ...] Python source  (null-terminated)
    SHELLCODE_OFF = 0
    PARAM_OFF     = 64
    CODE_OFF      = 104
    total_size    = CODE_OFF + len(INJECT_CODE)

    remote_mem = k32.VirtualAllocEx(
        h_proc, None, total_size, MEM_COMMIT_RESERVE, PAGE_EXECUTE_READWRITE,
    )
    if not remote_mem:
        print(f"ERROR: VirtualAllocEx failed — {k32.GetLastError()}")
        k32.CloseHandle(h_proc)
        return False

    base = ctypes.c_void_p(remote_mem).value
    shellcode_addr = base + SHELLCODE_OFF
    param_addr     = base + PARAM_OFF
    code_str_addr  = base + CODE_OFF

    param_struct = struct.pack("<QQQQQ",
        code_str_addr,   # [+0 ]  code_str_ptr
        remote_ensure,   # [+8 ]  PyGILState_Ensure
        remote_run,      # [+16]  PyRun_SimpleString
        remote_release,  # [+24]  PyGILState_Release
        0,               # [+32]  gilstate (written by shellcode)
    )

    payload = SHELLCODE + param_struct + INJECT_CODE
    assert len(payload) == total_size, f"{len(payload)} != {total_size}"

    written = ctypes.c_size_t(0)
    ok = k32.WriteProcessMemory(
        h_proc, ctypes.c_void_p(base),
        payload, total_size, ctypes.byref(written),
    )
    if not ok or written.value != total_size:
        print(f"ERROR: WriteProcessMemory failed — {k32.GetLastError()}")
        k32.VirtualFreeEx(h_proc, ctypes.c_void_p(base), 0, MEM_RELEASE)
        k32.CloseHandle(h_proc)
        return False

    print(f"Payload written to remote process at {hex(base)}")
    print("Creating remote thread …")

    h_thread = k32.CreateRemoteThread(
        h_proc, None, 0,
        ctypes.c_void_p(shellcode_addr),
        ctypes.c_void_p(param_addr),
        0, None,
    )
    if not h_thread:
        print(f"ERROR: CreateRemoteThread failed — {k32.GetLastError()}")
        k32.VirtualFreeEx(h_proc, ctypes.c_void_p(base), 0, MEM_RELEASE)
        k32.CloseHandle(h_proc)
        return False

    print("Remote thread running — waiting up to 120 s for model.save() …")
    k32.WaitForSingleObject(h_thread, 120_000)

    k32.CloseHandle(h_thread)
    k32.VirtualFreeEx(h_proc, ctypes.c_void_p(base), 0, MEM_RELEASE)
    k32.CloseHandle(h_proc)

    time.sleep(1)

    if os.path.exists(DONE_MARKER):
        with open(DONE_MARKER) as f:
            ts = f.read().strip()
        print(f"\nSUCCESS: checkpoint saved at timestep {ts}")
        print(f"  File: {SAVE_PATH}.zip")
        return True
    if os.path.exists(ERROR_MARKER):
        with open(ERROR_MARKER) as f:
            err = f.read()
        print(f"\nERROR in injected code:\n{err}")
        return False

    print("\nWARNING: no marker file found — injection may have timed out.")
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
