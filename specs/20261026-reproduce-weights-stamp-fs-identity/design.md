# Reproduce driver: weights stamps name the filesystem by its mount, not by `st_dev`

## The failure

Prefetch writes a stamp for each weights file it verifies: the blob's real path, the declared sha256 and the
blob's `lstat` size, `mtime_ns`, `st_dev` and `st_ino`. The inputs gate (`check_stamps`) re-reads `lstat` on the
host where the step lands and refuses any difference as `E_WEIGHTS_CHANGED`.

On a Slurm cluster whose model cache is on NFSv4, every GPU step that landed on a node other than the one that
ran prefetch failed `E_WEIGHTS_CHANGED` (10 of 25 engine-fit steps in one submission, all on the same other node),
with no file changed.

## The mechanism

An NFS client numbers each mount with an anonymous device of its own, so `st_dev` for the same file differs
between clients. Measured on two nodes for one directory on the project mount: `st_dev` 89 and 235, `st_ino`
9173602 on both, `statfs` `f_fsid` 0 on both (not usable either), and the mount's type and source
(`nfs4 nfs1:/group/proj`) the same on both. A scratch mount showed the same pattern (`st_dev` 92 and
101, same inode, same source).

## The change

- A stamp records `fs`, the filesystem identity, instead of `st_dev`: `filesystem_identity(path, st)` is the `<fstype>
  <source>` of the mount that holds the real path, from `/proc/self/mountinfo` (the mount whose mount point is the
  longest component-wise prefix; of mounts at the same point the last listed; octal escapes decoded). The mount's root
  is left out: the inode already names the file within its filesystem, and a host that mounts a subtree at the same path
  holds the same file. A later mount over a parent directory that hides a deeper mount is not detected (rare; it needs
  the mount tree). Where mountinfo is unavailable (not Linux) or covers nothing, it is `dev:<st_dev>`, which is what a
  single host compared before.
- `check_stamps` compares size, `mtime_ns`, inode and `fs`. A blob that changed, or that is now on another
  filesystem, is still `E_WEIGHTS_CHANGED`.
- `write_stamp`'s own check (did the file change while it was hashed, two `lstat` calls in one process) still
  compares `st_dev` as well: within one host it is meaningful.
- A stamp written before this change has no `fs`: the gate reports `E_WEIGHTS_UNVERIFIED`, and the next prefetch
  writes a new stamp (by the registered-input accept rule where it applies, otherwise by hashing).

## Tests

`src/tests/utils/reproduce/weights_stamp_fs_test.py`:

- mountinfo parsing (longest prefix on path components, shadowing, the escapes the kernel writes, optional fields, no
  parsable line);
- a stamp written on one host matches when every `lstat` reports another `st_dev` (red on the base: the gate
  reported `E_WEIGHTS_CHANGED`);
- a change is still caught under another `st_dev`;
- another mount source (a mountinfo file given to the test) is `E_WEIGHTS_CHANGED`, the same source read again
  is not; without readable mountinfo the identity is `dev:<st_dev>`;
- an old stamp without `fs` is `E_WEIGHTS_UNVERIFIED` until prefetch writes a new one;
- stamps record `fs` and no `st_dev`.
