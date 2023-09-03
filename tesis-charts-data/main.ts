import Thread from "https://deno.land/x/Thread/Thread.ts";

if (import.meta.main) {
  const thread = new Thread();
  const command = new Deno.Command(
    "/home/jorge/Documents/projects/tesisV4/venv/bin/python",
    {
      args: [
        "/home/jorge/Documents/projects/tesisV4/etc/e_model.py",
        "optimization",
        "--ts",
        "--wvp",
        "--e",
        "--gpu",
      ],
    },
  );
  const child = command.spawn();
  const { code, stdout, stderr } = command.outputSync();
  console.log(new TextDecoder().decode(stdout));
}

//   // open a file and pipe the subprocess output to it.
//   child.stdout.pipeTo(
//     Deno.openSync("output", { write: true, create: true }).writable,
//   );

//   // manually close stdin
//   child.stdin.close();
//   const status = await child.status;
//   console.log(status);
