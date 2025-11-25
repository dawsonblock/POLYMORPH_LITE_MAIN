import typer, os, getpass
from retrofitkit.compliance.users import Users
from retrofitkit.scripts.keygen import ensure_keys

app = typer.Typer(add_completion=False)

@app.command()
def init(admin_email: str = typer.Option(...), admin_name: str = typer.Option("Admin"), set_admin_password: bool = typer.Option(False)):
    ensure_keys()
    if set_admin_password:
        pw = getpass.getpass("Admin password: ")
    else:
        pw = "ChangeMeNow_123!"
    Users().create(admin_email, admin_name, "Admin", pw)
    typer.echo("Initialized. Admin created. Keys generated.")

if __name__ == "__main__":
    app()
