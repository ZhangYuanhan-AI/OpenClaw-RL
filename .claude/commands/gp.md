Run the following git commands in sequence:

1. `git add -A`
2. If "$ARGUMENTS" is "auto" or empty:
   - Run `git diff --cached` to see all staged changes
   - Automatically generate a concise, descriptive commit message in English based on the changes (e.g. "fix: adapt setup.sh for CUDA 12.2", "feat: add user auth endpoint", "chore: update dependencies")
   - Use the generated message for the commit
3. Otherwise, use "$ARGUMENTS" as the commit message directly
4. `git commit -m "<the commit message>"`
5. `git push`

If any step fails, stop and report the error. Do not use --no-verify or --force flags.