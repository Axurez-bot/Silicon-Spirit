"""
tools/audit_analyzer.py
Advanced dual-phase security laboratory.
Executes deep static AST auditing coupled with dynamic ephemeral container sandboxing.
"""
import os
import ast
import json
import time
from pathlib import Path
from crewai.tools import tool

class AdvancedASTAuditor(ast.NodeVisitor):
    def __init__(self):
        self.findings = []
        self.risk_score = 0

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                self.findings.append({
                    "type": "Critical Execution Primitive",
                    "details": f"Direct use of dangerous built-in '{node.func.id}'.",
                    "line": node.lineno,
                    "severity": "CRITICAL"
                })
                self.risk_score += 25

            if node.func.id in ['getattr', 'setattr']:
                self.findings.append({
                    "type": "Reflection/Introspection Hook",
                    "details": f"Dynamic attribute lookup via '{node.func.id}'. Potential sandbox evasion.",
                    "line": node.lineno,
                    "severity": "HIGH"
                })
                self.risk_score += 15

        if isinstance(node.func, ast.Name) and node.func.id == '__import__':
            if node.args and isinstance(node.args[0], ast.BinOp) and isinstance(node.args[0].op, ast.Add):
                self.findings.append({
                    "type": "Obfuscated Import",
                    "details": "String concatenation detected inside an __import__ statement.",
                    "line": node.lineno,
                    "severity": "HIGH"
                })
                self.risk_score += 20

        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in ['os', 'subprocess', 'sys', 'ctypes', 'socket', 'urllib', 'requests']:
                self.findings.append({
                    "type": "Dangerous Module Import",
                    "details": f"Imported highly sensitive OS/Network module: '{alias.name}'.",
                    "line": node.lineno,
                    "severity": "MEDIUM"
                })
                self.risk_score += 10
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module in ['os', 'subprocess', 'sys', 'ctypes', 'socket']:
            self.findings.append({
                "type": "Dangerous Submodule Import",
                "details": f"Imported members directly from sensitive module: '{node.module}'.",
                "line": node.lineno,
                "severity": "MEDIUM"
            })
            self.risk_score += 10
        self.generic_visit(node)


@tool("advanced_threat_audit")
def advanced_threat_audit(filename: str) -> str:
    """
    Performs deep static AST structural auditing on staged script targets.
    If the code profile is clean enough, spins up an isolated, network-less 
    ephemeral Docker container dynamically to parse runtime execution metrics securely.
    """
    sandbox_dir = Path("/app/spirit_memory/analysis_sandbox")
    target_path = (sandbox_dir / Path(filename).name).resolve()
    
    if not str(target_path).startswith(str(sandbox_dir)):
        return json.dumps({"status": "BLOCKED", "reason": "Path traversal attempt caught."})
        
    if not target_path.exists():
        return json.dumps({"status": "ERROR", "reason": f"Staged target '{filename}' not found."})
        
    try:
        source_code = target_path.read_text(errors='ignore')
        
        # Phase 1: Structural AST Compilation Audit
        tree = ast.parse(source_code)
        auditor = AdvancedASTAuditor()
        auditor.visit(tree)
        
        risk_level = "LOW"
        if auditor.risk_score >= 50: risk_level = "CRITICAL"
        elif auditor.risk_score >= 30: risk_level = "HIGH"
        elif auditor.risk_score >= 15: risk_level = "MEDIUM"
        
        # Phase 2: Dynamic Ephemeral Docker Sandbox Execution
        execution_output = "Skipped Execution (Static risk tier too high or file unsupported)"
        
        # Only attempt containerization execution if the script is not critically dangerous
        if auditor.risk_score < 50 and target_path.suffix == '.py':
            try:
                import docker
                client = docker.from_env()
                
                # Setup volume configuration mapping to share just the target code file
                volumes_spec = {
                    str(target_path): {
                        'bind': '/scratch/target.py',
                        'mode': 'ro'
                    }
                }
                
                # Provision the disposable run environment
                container = client.containers.run(
                    image="python:3.10-slim",
                    command=["python", "/scratch/target.py"],
                    volumes=volumes_spec,
                    network_mode="none",          # COMPLETELY CUT NETWORK ACCESS
                    mem_limit="128m",             # Prevent memory leaks/bombs
                    nano_cpus=500000000,          # Cap at maximum 0.5 CPU core utilization
                    detach=True
                )
                
                # Enforce a definitive execution timeout limit loop
                timeout = 10
                start_time = time.time()
                while container.status != 'exited':
                    time.sleep(0.5)
                    container.reload()
                    if time.time() - start_time > timeout:
                        container.kill()
                        execution_output = "Execution killed: Container sandbox time limit exceeded (Possible loop/hang)."
                        risk_level = "HIGH"
                        break
                
                if "killed" not in execution_output:
                    # Capture pristine isolation standard logs
                    logs = container.logs(stdout=True, stderr=True).decode('utf-8')
                    execution_output = logs if logs.strip() else "[Execution finished with zero output strings]"
                
                # Complete disposal cleanup
                container.remove(force=True)
                
            except ImportError:
                execution_output = "Dynamic sandbox unavailable: 'docker' library python bindings not installed inside environment."
            except Exception as docker_err:
                execution_output = f"Dynamic execution subsystem error: {str(docker_err)}"

        report = {
            "target": filename,
            "metrics": {
                "total_lines": len(source_code.splitlines()),
                "raw_character_count": len(source_code),
                "aggregated_risk_score": auditor.risk_score,
                "threat_tier": risk_level
            },
            "findings": auditor.findings,
            "dynamic_sandbox_execution": execution_output,
            "remediation_plan": (
                f"1. Analyze the identified findings: {auditor.findings}. "
                f"2. Propose the exact, non-vulnerable code replacement structure for '{filename}'. "
                f"3. Output a clean, safe patch block avoiding dynamic primitives."
            ),
            "status": "COMPLETED"
        }
        
        return json.dumps(report, indent=4)
        
    except SyntaxError as se:
        return json.dumps({
            "status": "FAILED", 
            "reason": f"Malformed script syntax. Structural compilation failed: {str(se)}"
        })
    except Exception as e:
        return json.dumps({"status": "ERROR", "reason": f"Runtime auditing crash: {str(e)}"})