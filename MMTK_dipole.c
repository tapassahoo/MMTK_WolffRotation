/* C routines for dipoleFF.py */

#include "MMTK/universe.h"
#include "MMTK/forcefield.h"
#include "MMTK/forcefield_private.h"
#include <math.h>

/* This function does the actual energy (and gradient) calculation.
   Everything else is just bookkeeping. */
static void
dipole_evaluator(PyFFEnergyTermObject *self,
		   PyFFEvaluatorObject *eval,
		   energy_spec *input,
		   energy_data *energy)
     /* The four parameters are pointers to structures that are
	defined in MMTK/forcefield.h.
	PyFFEnergyTermObject: All data relevant to this particular
                              energy term.
        PyFFEvaluatorObject:  Data referring to the global energy
                              evaluation process, e.g. parallelization
                              options. Not used here.
        energy_spec:          Input parameters for this routine, i.e.
                              atom positions and parallelization parameters.
        energy_data:          Storage for the results (energy terms,
                              gradients, second derivatives).
     */
{
  vector3 *coordinates = (vector3 *)input->coordinates->data;
  vector3 *g;
  int atom_index1 = (int)self->param[0];  /* atom index */
  int atom_index2 = (int)self->param[1];  /* atom index */
  int atom_index3 = (int)self->param[2];  /* atom index */
  int atom_index4 = (int)self->param[3];  /* atom index */

/*
  printf("%i %i %i %i %i %i \n",atom_index1,atom_index2,atom_index3,atom_index4,atom_index5,atom_index6);
*/

  double x1 = coordinates[atom_index1][0];
  double y1 = coordinates[atom_index1][1];
  double z1 = coordinates[atom_index1][2];

  double x2 = coordinates[atom_index2][0];
  double y2 = coordinates[atom_index2][1];
  double z2 = coordinates[atom_index2][2];

  double x3 = coordinates[atom_index3][0];
  double y3 = coordinates[atom_index3][1];
  double z3 = coordinates[atom_index3][2];

  double x4 = coordinates[atom_index4][0];
  double y4 = coordinates[atom_index4][1];
  double z4 = coordinates[atom_index4][2];

  /* energy_terms is an array because each routine could compute
     several terms that should logically be kept apart. For example,
     a single routine calculates Lennard-Jones and electrostatic interactions
     in a single iteration over the nonbonded list. The separation of
     terms is only done for the benefit of user code (universe.energyTerms())
     returns the value of each term separately), the total energy is
     always the sum of all terms. Here we have only one energy term. */

  // unit convertion to MMTK units
  double massF=18.998403;
  double massH=1.00794;
  double e,dipolemoment,bondlength,r,charge;
  double massFpercent, massHpercent, mu1dotmu2,mu1dotr,mu2dotr;
  double gr[12];
  double mu1[3];
  double mu2[3];
  double cmvec[3];

  dipolemoment=2.0; //Dipole moment (Debye)
  bondlength=0.09255215; //Bond Length (nm)
  charge=(dipolemoment/(bondlength*1.0e-9))*3.33564e-30; //Convert to (Coulomb)

  double q=charge;

  mu1[0]=(x2-x1);
  mu1[1]=(y2-y1);
  mu1[2]=(z2-z1);

  mu2[0]=(x4-x3);
  mu2[1]=(y4-y3);
  mu2[2]=(z4-z3);


  massFpercent=massF/(massF+massH);
  massHpercent=massH/(massF+massH);

  cmvec[0]=(massFpercent*(x3-x1)+massHpercent*(x4-x2));
  cmvec[1]=(massFpercent*(y3-y1)+massHpercent*(y4-y2));
  cmvec[2]=(massFpercent*(z3-z1)+massHpercent*(z4-z2));

  r=sqrt(cmvec[0]*cmvec[0]+cmvec[1]*cmvec[1]+cmvec[2]*cmvec[2]);

  mu1dotmu2=mu1[0]*mu2[0]+mu1[1]*mu2[1]+mu1[2]*mu2[2];
  mu1dotr=mu1[0]*cmvec[0]+mu1[1]*cmvec[1]+mu1[2]*cmvec[2];
  mu2dotr=mu2[0]*cmvec[0]+mu2[1]*cmvec[1]+mu2[2]*cmvec[2];

  e=(q*q/(4.0*M_PI*r*r*r*8.854187817e-12))*(mu1dotmu2-3.0*mu1dotr*mu2dotr/(r*r))*(1.0e9*6.022140857e23)/1000.0;

  double predv=(q*q/(4.0*M_PI*r*r*r*8.854187817e-12))*(1.0e9*6.022140857e23)/1000.0;
  // components
  gr[0]=3.0*predv*cmvec[0]*massFpercent*mu1dotmu2/(r*r)-predv*mu2[0]
    -15.0*(predv/(r*r*r*r))*cmvec[0]*massFpercent*mu1dotr*mu2dotr
    -3.0*(predv/(r*r))*(-1.0*cmvec[0]-massFpercent*mu1[0])*mu2dotr
    +3.0*(predv/(r*r))*mu1dotr*massFpercent*mu2[0];

  gr[1]=3.0*predv*cmvec[1]*massFpercent*mu1dotmu2/(r*r)-predv*mu2[1]
    -15.0*(predv/(r*r*r*r))*cmvec[1]*massFpercent*mu1dotr*mu2dotr
    -3.0*(predv/(r*r))*(-1.0*cmvec[1]-massFpercent*mu1[1])*mu2dotr
    +3.0*(predv/(r*r))*mu1dotr*massFpercent*mu2[1];

  gr[2]=3.0*predv*cmvec[2]*massFpercent*mu1dotmu2/(r*r)-predv*mu2[2]
    -15.0*(predv/(r*r*r*r))*cmvec[2]*massFpercent*mu1dotr*mu2dotr
    -3.0*(predv/(r*r))*(-1.0*cmvec[2]-massFpercent*mu1[2])*mu2dotr
    +3.0*(predv/(r*r))*mu1dotr*massFpercent*mu2[2];

  gr[3]=3.0*predv*cmvec[0]*massHpercent*mu1dotmu2/(r*r)+predv*mu2[0]
    -15.0*(predv/(r*r*r*r))*cmvec[0]*massHpercent*mu1dotr*mu2dotr
    -3.0*(predv/(r*r))*(cmvec[0]-massHpercent*mu1[0])*mu2dotr
    +3.0*(predv/(r*r))*mu1dotr*massHpercent*mu2[0];

  gr[4]=3.0*predv*cmvec[1]*massHpercent*mu1dotmu2/(r*r)+predv*mu2[1]
    -15.0*(predv/(r*r*r*r))*cmvec[1]*massHpercent*mu1dotr*mu2dotr
    -3.0*(predv/(r*r))*(cmvec[1]-massHpercent*mu1[1])*mu2dotr
    +3.0*(predv/(r*r))*mu1dotr*massHpercent*mu2[1];

  gr[5]=3.0*predv*cmvec[2]*massHpercent*mu1dotmu2/(r*r)+predv*mu2[2]
    -15.0*(predv/(r*r*r*r))*cmvec[2]*massHpercent*mu1dotr*mu2dotr
    -3.0*(predv/(r*r))*(cmvec[2]-massHpercent*mu1[2])*mu2dotr
    +3.0*(predv/(r*r))*mu1dotr*massHpercent*mu2[2];

  gr[6]=-3.0*predv*cmvec[0]*massFpercent*mu1dotmu2/(r*r)-predv*mu1[0]
    +15.0*(predv/(r*r*r*r))*cmvec[0]*massFpercent*mu1dotr*mu2dotr
    -3.0*(predv/(r*r))*mu1[0]*massFpercent*mu2dotr
    -3.0*(predv/(r*r))*mu1dotr*(-1.0*cmvec[0]+mu2[0]*massFpercent);

  gr[7]=-3.0*predv*cmvec[1]*massFpercent*mu1dotmu2/(r*r)-predv*mu1[1]
    +15.0*(predv/(r*r*r*r))*cmvec[1]*massFpercent*mu1dotr*mu2dotr
    -3.0*(predv/(r*r))*mu1[1]*massFpercent*mu2dotr
    -3.0*(predv/(r*r))*mu1dotr*(-1.0*cmvec[1]+mu2[1]*massFpercent);

  gr[8]=-3.0*predv*cmvec[2]*massFpercent*mu1dotmu2/(r*r)-predv*mu1[2]
    +15.0*(predv/(r*r*r*r))*cmvec[2]*massFpercent*mu1dotr*mu2dotr
    -3.0*(predv/(r*r))*mu1[2]*massFpercent*mu2dotr
    -3.0*(predv/(r*r))*mu1dotr*(-1.0*cmvec[2]+mu2[2]*massFpercent);

  gr[9] =-3.0*predv*cmvec[0]*massHpercent*mu1dotmu2/(r*r)+predv*mu1[0]
    +15.0*(predv/(r*r*r*r))*cmvec[0]*massHpercent*mu1dotr*mu2dotr
    -3.0*(predv/(r*r))*mu1[0]*massHpercent*mu2dotr
    -3.0*(predv/(r*r))*mu1dotr*(cmvec[0]+mu2[0]*massHpercent);

  gr[10]=-3.0*predv*cmvec[1]*massHpercent*mu1dotmu2/(r*r)+predv*mu1[1]
    +15.0*(predv/(r*r*r*r))*cmvec[1]*massHpercent*mu1dotr*mu2dotr
    -3.0*(predv/(r*r))*mu1[1]*massHpercent*mu2dotr
    -3.0*(predv/(r*r))*mu1dotr*(cmvec[1]+mu2[1]*massHpercent);

  gr[11]=-3.0*predv*cmvec[2]*massHpercent*mu1dotmu2/(r*r)+predv*mu1[2]
    +15.0*(predv/(r*r*r*r))*cmvec[2]*massHpercent*mu1dotr*mu2dotr
    -3.0*(predv/(r*r))*mu1[2]*massHpercent*mu2dotr
    -3.0*(predv/(r*r))*mu1dotr*(cmvec[2]+mu2[2]*massHpercent);

  energy->energy_terms[self->index] = e;
  /* If only the energy is asked for, stop here. */
  if (energy->gradients == NULL)
    return;

  /* Add the gradient contribution to the global gradient array.
     It would be a serious error to use '=' instead of '+=' here,
     in that case all previously calculated forces would be erased. */

  g = (vector3 *)((PyArrayObject*)energy->gradients)->data;

  g[atom_index1][0]+=gr[0];
  g[atom_index1][1]+=gr[1];
  g[atom_index1][2]+=gr[2];

  g[atom_index2][0]+=gr[3];
  g[atom_index2][1]+=gr[4];
  g[atom_index2][2]+=gr[5];

  g[atom_index3][0]+=gr[6];
  g[atom_index3][1]+=gr[7];
  g[atom_index3][2]+=gr[8];

  g[atom_index4][0]+=gr[9];
  g[atom_index4][1]+=gr[10];
  g[atom_index4][2]+=gr[11];

}

/* A utility function that allocates memory for a copy of a string */
static char *
allocstring(char *string)
{
  char *memory = (char *)malloc(strlen(string)+1);
  if (memory != NULL)
    strcpy(memory, string);
  return memory;
}

/* The next function is meant to be called from Python. It creates the
   energy term object at the C level and stores all the parameters in
   there in a form that is convient to access for the C routine above.
   This is the routine that is imported into and called by the Python
   module, HeHeFF.py. */
static PyObject *
dipoleTerm(PyObject *dummy, PyObject *args)
{
  PyFFEnergyTermObject *self;
  int atom_index1;
  int atom_index2;
  int atom_index3;
  int atom_index4;

  /* Create a new energy term object and return if the creation fails. */
  self = PyFFEnergyTerm_New();
  if (self == NULL)
    return NULL;
  /* Convert the parameters to C data types. */
  if (!PyArg_ParseTuple(args, "O!iiii",
			&PyUniverseSpec_Type, &self->universe_spec,
			&atom_index1, &atom_index2, &atom_index3, &atom_index4))
    return NULL;
  /* We keep a reference to the universe_spec in the newly created
     energy term object, so we have to increase the reference count. */
  Py_INCREF(self->universe_spec);
  /* A pointer to the evaluation routine. */
  self->eval_func = dipole_evaluator;
  /* The name of the energy term object. */
  self->evaluator_name = "dipole";
  /* The names of the individual energy terms - just one here. */
  self->term_names[0] = allocstring("dipole");
  if (self->term_names[0] == NULL)
    return PyErr_NoMemory();
  self->nterms = 1;
  /* self->param is a storage area for parameters. Note that there
     are only 40 slots (double) there, if you need more space, you can use
     self->data, an array for up to 40 Python object pointers. */
  self->param[0] = (double) atom_index1;
  self->param[1] = (double) atom_index2;
  self->param[2] = (double) atom_index3;
  self->param[3] = (double) atom_index4;
//  self->param[1] = (double) atom_index2;
  /* Return the energy term object. */
  return (PyObject *)self;
}

/* This is a list of all Python-callable functions defined in this
   module. Each list entry consists of the name of the function object
   in the module, the C routine that implements it, and a "1" signalling
   new-style parameter passing conventions (only veterans care about the
   alternatives). The list is terminated by a NULL entry. */
static PyMethodDef functions[] = {
  {"dipoleTerm", dipoleTerm, 1},
  {NULL, NULL}		/* sentinel */
};


/* The initialization function for the module. This is the only function
   that must be publicly visible, everything else should be declared
   static to prevent name clashes with other modules. The name of this
   function must be "init" followed by the module name. */
DL_EXPORT(void)
initMMTK_dipole(void)
{
  PyObject *m;

  /* Create the module and add the functions. */
  m = Py_InitModule("MMTK_dipole", functions);

  /* Import the array module. */
#ifdef import_array
  import_array();
#endif

  /* Import MMTK modules. */
  import_MMTK_universe();
  import_MMTK_forcefield();

  /* Check for errors. */
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module MMTK_dipole");
}
