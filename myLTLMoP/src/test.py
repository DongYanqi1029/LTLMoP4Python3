from lib.execute import execute_main
import logging
import sys, os, getopt, textwrap
import threading, subprocess, time
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import xmlrpc.client
import socket
from lib.execute import LTLMoPExecutor

def execute_strat(listen_port=None, spec_file=None, aut_file=None, show_gui=False):
    logging.info("Hello. Let's do this!")

    # Create the XML-RPC server
    if listen_port is None:
        # Search for a port we can successfully bind to
        while True:
            listen_port = random.randint(10000, 65535)
            try:
                xmlrpc_server = SimpleXMLRPCServer(("127.0.0.1", listen_port), logRequests=False, allow_none=True)
            except socket.error as e:
                pass
            else:
                break
    else:
        xmlrpc_server = SimpleXMLRPCServer(("127.0.0.1", listen_port), logRequests=False, allow_none=True)

    # Create the execution context object
    e = LTLMoPExecutor()

    # Register functions with the XML-RPC server
    xmlrpc_server.register_instance(e)

    # Kick off the XML-RPC server thread
    XMLRPCServerThread = threading.Thread(target=xmlrpc_server.serve_forever)
    XMLRPCServerThread.daemon = True
    XMLRPCServerThread.start()
    logging.info("Strategy Executor listening for XML-RPC calls on http://127.0.0.1:{} ...".format(listen_port))

    # Start the GUI if necessary
    if show_gui:
        # Create a subprocess
        logging.info("Starting GUI window...")
        p_gui = subprocess.Popen([sys.executable, "-u", "-m", "lib.simGUI", str(listen_port)])

        # Wait for GUI to fully load, to make sure that
        # to make sure all messages are redirected
        e.externalEventTargetRegistered.wait()

    if spec_file is not None:
        # Tell executor to load spec & aut
        #if aut_file is None:
        #    aut_file = spec_file.rpartition('.')[0] + ".aut"
        e.initialize(spec_file, aut_file, firstRun=True)

    # Start the executor's main loop in this thread
    e.run()

    # Clean up on exit
    logging.info("Waiting for XML-RPC server to shut down...")
    xmlrpc_server.shutdown()
    XMLRPCServerThread.join()
    logging.info("XML-RPC server shutdown complete.  Goodbye.")

if __name__ == "__main__":
    ### Check command-line arguments

    logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                    datefmt=' %Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)

    aut_file = None
    spec_file = None
    show_gui = True
    listen_port = None

    # import sys
    # print(sys.modules)

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hnp:a:s:", ["help", "no-gui", "xmlrpc-listen-port=", "aut-file=", "spec-file="])
    except getopt.GetoptError:
        logging.exception("Bad arguments")
        usage(sys.argv[0])
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage(sys.argv[0])
            sys.exit()
        elif opt in ("-n", "--no-gui"):
            show_gui = False
        elif opt in ("-p", "--xmlrpc-listen-port"):
            try:
                listen_port = int(arg)
            except ValueError:
                logging.error("Invalid port '{}'".format(arg))
                sys.exit(2)
        elif opt in ("-a", "--aut-file"):
            aut_file = arg
        elif opt in ("-s", "--spec-file"):
            spec_file = arg

    execute_main(listen_port, spec_file, aut_file, show_gui)